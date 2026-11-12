from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import savgol_filter
from six.moves import xrange
# import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import time
import os
from classifier import CNN
from unexpressed import calcul_A_MNIST
from VAE_model import VAE
from Diffusion_model import SimpleUnet, linear_beta_schedule, forward_diffusion_sample, denoise2
from QAM import Ps_qam, snr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def add_noise(data, snr_idx):
    if snr_idx<len(Ps_qam):
        ber = Ps_qam[snr_idx]
        num_dim = data.numel()
        erase_indices = np.random.choice(num_dim, size=int(num_dim*ber), replace=False)
        data_flat = data.view(-1)
        data_flat[erase_indices] = 0
        data_return = data_flat.view(data.size())
    else:
        data_return = data
    return data_return




if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])) #, transforms.Normalize((0.1307), (0.3081))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])) #, transforms.Normalize((0.1307), (0.3081))
    # data_variance = np.var(training_data.data / 255.0)  # 0.06328692405746414
    # test_data_variance = np.var(validation_data.data / 255.0)  # 0.06311123249672204

    def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    set_seed(0)
    batch_size = 100
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True)




    # Parameters setting
    num_epochs = 200
    num_warmup_epochs = 150
    x_dim = training_data.data.shape[1]**2
    hidden_dim = 256
    encoder_dim = [-1, 1, 16, 16]
    learning_rate = 1e-3
    encoder_method = 'VAE_256'
    snrs=[]
    method= 'Diffusion_denoising__RT_denoising'
    train = True
    model_method = 'Diffusion_denoising'
    train_noise= True
    adapt_noise= True



    # Load model
    model = SimpleUnet().to(device)
    encoder = VAE(x_dim, hidden_dim).to(device)
    encoder.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + encoder_method + '.pth'))
    for param in encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # Define beta schedule
    T = 300
    betas = linear_beta_schedule(timesteps=T)
    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    reconst_loss_train, noise_loss_train = [], []
    reconst_loss_test, noise_loss_test, semantic_loss_test = [], [], []
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))

    for e in range(num_warmup_epochs+num_epochs):
        reconst_loss_tr, noise_loss_tr = 0.0, 0.0
        reconst_loss_ts, noise_loss_ts, semantic_loss_ts = 0.0, 0.0, 0.0

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            model.train()
            x = x.view(-1, 1, 28,28)
            x = x.to(device)

            if train==True:
                optimizer.zero_grad()
                t = torch.randint(0, T, (batch_size,), device=device).long()

                mu, _ = encoder.encode(x.view(-1, x_dim))
                z = mu.view(encoder_dim)

                #TODO block bits in mu, logvar
                if train_noise == True:
                    ber_index = random.randint(1, len(snr)+1)
                    z = add_noise(z, ber_index-1)

                x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
                noise_pred = model(x_noisy, t)

                if e < num_warmup_epochs:
                    loss = nn.MSELoss()(noise, noise_pred) #F.l1_loss
                    noise_loss_tr += loss.item()
                else:
                    recon_x = denoise2(x_noisy, T, noise_pred)
                    loss = nn.MSELoss()(recon_x, x)
                    reconst_loss_tr += loss.item()


                loss.backward()
                optimizer.step()
            else:
                model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + model_method + '.pth'))



            #TODO calculate A
            if e == num_warmup_epochs+num_epochs-1 :
                print('saving A for batch: ', batch_idx_tr)

                t = torch.randint(0, T, (batch_size,), device=device).long()
                mu, _ = encoder.encode(x.view(-1, x_dim))
                z = mu.view(encoder_dim)
                if adapt_noise == True:
                    ber_index = random.randint(1, len(snr) + 1)
                    z = add_noise(z, ber_index-1)
                x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
                noise_pred = model(x_noisy, t)
                recon_x = denoise2(x_noisy, T, noise_pred)

                for d in np.arange(2, 28, 2):
                    x, recon_x = x.cpu(), recon_x.cpu()
                    if '_A' in method:
                        # smalleigen
                        A = calcul_A_MNIST(x, recon_x, d, False).cpu()
                    else:
                        A = calcul_A_MNIST(x, recon_x, d, True).cpu()  # (d, 28)
                    try:
                        torch.save(A, f'As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')
                    except:
                        os.makedirs(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}')
                        torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')




        with torch.no_grad():
            if train==True:
                for batch_idx_ts, (x, label) in enumerate(validation_loader):
                    x = x.view(-1, 1, 28, 28)
                    x, label = x.to(device), label.to(device)
                    t = torch.randint(0, T, (batch_size,), device=device).long()
                    mu, _ = encoder.encode(x.view(-1, x_dim))
                    z = mu.view(encoder_dim)

                    # TODO noise add to mu, logvar
                    ber_index = random.randint(1, len(snr) + 1)
                    z = add_noise(z, ber_index - 1)

                    x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
                    noise_pred = model(x_noisy, t)
                    recon_x = denoise2(x_noisy, T, noise_pred)

                    noise_loss = F.l1_loss(noise, noise_pred)
                    reconst_loss = nn.MSELoss()(recon_x, x)
                    classifier_loss = nn.CrossEntropyLoss()(classifier(recon_x.view(batch_size, 1, 28, 28)), label)

                    reconst_loss_ts += reconst_loss.item()
                    noise_loss_ts += noise_loss.item()
                    semantic_loss_ts += classifier_loss.item()


        reconst_loss_train.append(reconst_loss_tr / batch_idx_tr)
        noise_loss_train.append(noise_loss_tr / batch_idx_tr)
        reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
        noise_loss_test.append(noise_loss_ts / batch_idx_ts)
        semantic_loss_test.append(semantic_loss_ts / batch_idx_ts)

        #
        print(f'[{e} / {num_epochs}] reconstruction loss: {reconst_loss_test[-1]}, regularization loss: {noise_loss_test[-1]}, semantic loss: {semantic_loss_test[-1]}')


        if train == True:
            # np.save('results/reconst_loss_train_'+ method + '.npy', reconst_loss_train)
            # np.save('results/regular_loss_train_' + method + '.npy', regular_loss_train)
            np.save('results/reconst_loss_test_' + method + '.npy', reconst_loss_test)
            np.save('results/noise_loss_test_' + method + '.npy', noise_loss_test)
            np.save('results/semantic_loss_test_' + method + '.npy', semantic_loss_test)

        if train==True:
            torch.save(model.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/'+model_method+'.pth')


