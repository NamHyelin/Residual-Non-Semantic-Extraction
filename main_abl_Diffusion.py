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
from unexpressed import calcul_A_MNIST, calcul_A_eigenvalues, calcul_A_singular_MNIST, calcul_A_singularvalues
from QAM import snr, Ps_qam
from SNR_Diffusion import add_noise
from VAE_model import VAE
from Diffusion_model import forward_diffusion_sample, SimpleUnet, denoise2, linear_beta_schedule
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")









def test_A_forsample(batchsize, save_method):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses=np.zeros((14))
        for batch_idx, (x_, y) in enumerate(training_loader):
            x = x_.view(-1, 1, 28, 28)
            x = x.to(device)

            t = torch.randint(0, T, (batch_size,), device=device).long()

            mu, _ = encoder.encode(x.view(-1,x_dim))
            z = mu.view(encoder_dim)
            x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
            noise_pred = model(x_noisy, t)
            recon_x = denoise2(x_noisy, T, noise_pred)
            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()


            for batch in range(int(batch_size/batchsize)):
                #TODO plot under-expessed term
                d_loss = np.zeros((14))
                x_b = x[batch * batchsize: (batch + 1) * batchsize]
                recon_x_b = recon_x[batch * batchsize: (batch + 1) * batchsize]
                d_loss[0] += nn.MSELoss()(recon_x_b.view(-1, 28, 28), x_b.view(-1, 28, 28)).item() / batchsize

                for d in np.arange(2, 28, 2):
                    A = calcul_A_MNIST(x_b, recon_x_b, d, True).to(device)
                    # A = calcul_A_singular_MNIST(x, recon_x, 28, d, True).to(device)

                    Ax = torch.matmul(A, x.view(-1,28,28))
                    A_ = torch.linalg.pinv(A)
                    if 'Ax' in save_method:
                        final_recon_x_b = torch.matmul(A_, torch.matmul(A, x_b.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x_b.view(-1, 28, 28))
                    else:
                        final_recon_x_b = torch.matmul(torch.matmul(A_, A), x_b.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x_b.view(-1, 28, 28))

                    final_reconst_loss = nn.MSELoss()(final_recon_x_b, x_b.view(-1,28,28)).item()
                    d_loss[int(d/2)]+=(final_reconst_loss/batchsize)
                d_losses += d_loss

                if batch_size*batch_idx > 10000:
                    break


        if type(batchsize) == str:
            np.save(f'results/final_mse_loss_{save_method}.npy', d_losses/(batch_idx))
        else:
            np.save(f'results/final_mse_loss_{save_method}.npy', d_losses / (batch_idx))

        print('final losses wrt d: ', d_losses/batch_idx)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')
        plt.show()




def test_A_forsample_SNR(batchsize, save_method):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses = np.zeros((len(snr)+1, len(np.arange(2, 28, 2))+1))
        for batch_idx, (x_, y) in enumerate(training_loader):
            print('Batch: ', batch_idx)
            x = x.view(-1, 1, 28, 28)
            x = x.to(device)

            t = torch.randint(0, T, (batch_size,), device=device).long()
            mu, _ = encoder.encode(x.view(-1, x_dim))
            z = mu.view(encoder_dim)

            for ber_index in np.arange(1, len(snr)+2):
                z = add_noise(z, ber_index-1)

                x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
                noise_pred = model(x_noisy, t)
                recon_x = denoise2(x_noisy, T, noise_pred)

                reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
                d_losses[ber_index - 1, 0] += reconst_loss / batch_size


                #TODO plot under-expessed term
                for d in np.arange(2, 28, 2):
                    A = calcul_A_MNIST(x, recon_x, d, True).to(device)

                    Ax = torch.matmul(A, x.view(-1,28,28))
                    Ax = add_noise(Ax, ber_index-1)

                    A_ = torch.linalg.pinv(A)
                    final_recon_x = torch.matmul(A_, Ax) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))


                    final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()

                    d_losses[ber_index - 1, int(d / 2)] += final_reconst_loss / batch_size


        if type(batchsize) == str:
            np.save(f'results/final_mse_loss_{save_method}npy', d_losses/batch_idx)
        else:
            np.save(f'results/final_mse_loss_{save_method}.npy', d_losses / batch_idx)
        print('original loss: ', origin_mse_loss/batch_idx, '\n')
        print('final losses wrt d: ', d_losses/batch_idx)







def test_A_forsample_singular(batchsize, save_method):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        m_losses=np.zeros((16))
        Ax_sizes = np.zeros((15,3))
        A_sizes = np.zeros((15,2))
        null_losses = np.zeros((15))
        null_x_losses = np.zeros((15))
        range_losses = np.zeros((15))
        for batch_idx, (x_, y) in enumerate(training_loader):
            x = x.view(-1, 1, 28, 28)
            x = x.to(device)

            t = torch.randint(0, T, (batch_size,), device=device).long()

            mu, _ = encoder.encode(x.view(-1, x_dim))
            z = mu.view(encoder_dim)
            x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
            noise_pred = model(x_noisy, t)
            recon_x = denoise2(x_noisy, T, noise_pred)

            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()


            #TODO plot under-expessed term
            m_loss = []
            m_loss.append(reconst_loss/batch_size)

            mc=0
            for m in [1,2,4,7,8,14,16,28,49,56,98,112,196,392,784]:
                d = calcul_A_singularvalues(x, recon_x, m)
                A = calcul_A_singular_MNIST(x, recon_x, m, d, True).to(device)

                Ax = torch.matmul(A, x.view(-1, m, int(784/m)))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, m, int(784/m)))) + torch.matmul((torch.eye(m).to(device) - torch.matmul(A_, A)), recon_x.view(-1, m, int(784/m)))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, m, int(784/m))) + torch.matmul((torch.eye(m).to(device) - torch.matmul(A_, A)), recon_x.view(-1, m, int(784/m)))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,m, int(784/m))).item()

                m_loss.append(final_reconst_loss/batch_size)
                range_losses[mc] += nn.MSELoss()(torch.matmul(torch.matmul(A_, A), x.view(-1, m, int(784/m))), x.view(-1,m, int(784/m))).item()/batchsize
                null_losses[mc] += nn.MSELoss()(torch.matmul((torch.eye(m).to(device) - torch.matmul(A_, A)), recon_x.view(-1, m, int(784/m))), recon_x.view(-1, m, int(784/m))).item()/batchsize
                null_x_losses[mc] += nn.MSELoss()(torch.matmul((torch.eye(m).to(device) - torch.matmul(A_, A)), recon_x.view(-1, m, int(784 / m))), x.view(-1, m, int(784 / m))).item() / batchsize
                Ax_sizes[mc] += Ax.size()
                A_sizes[mc] += A.size()
                mc+=1
            m_losses += m_loss

            if batch_idx*batchsize==60000:
                break
            print(f'Batch idx: ', batch_idx)


        if batch_idx==0: batch_idx=1
        if type(batchsize) == str:
            np.save(f'results/final_mse_loss_{save_method}.npy', m_losses/batch_idx)
        else:
            np.save(f'results/final_mse_loss_{save_method}.npy', m_losses / batch_idx)
        np.save(f'results/Ax_sizes_{save_method}.npy', Ax_sizes / (batch_idx))
        np.save(f'results/A_sizes_{save_method}.npy', A_sizes / (batch_idx))

        np.save(f'results/range_losses_{save_method}.npy', range_losses / (batch_idx))
        np.save(f'results/null_losses_{save_method}.npy', null_losses / (batch_idx))
        np.save(f'results/null_x_losses_{save_method}.npy', null_x_losses / (batch_idx))


        print('final losses wrt d: ', m_losses/batch_idx)
        plt.plot(m_losses/batch_idx)
        plt.xticks(np.arange(0, 16, 1), np.arange(0, 16, 1))
        plt.xlabel('m')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')
        plt.show()







def test_A_forsample_oneimg(save_method):
    with torch.no_grad():
        final_losses = []
        initial_losses =[]
        for batch_idx, (x_, y) in enumerate(training_loader):
            for imgidx in range(batch_size):
                x = x.view(-1, 1, 28, 28)
                x = x.to(device)

                t = torch.randint(0, T, (batch_size,), device=device).long()

                mu, _ = encoder.encode(x.view(-1, x_dim))
                z = mu.view(encoder_dim)
                x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
                noise_pred = model(x_noisy, t)
                recon_x = denoise2(x_noisy, T, noise_pred)

                reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()


                #TODO plot under-expessed term
                d = calcul_A_eigenvalues(x, recon_x)
                A = calcul_A_MNIST(x, recon_x, d, True).to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()

                final_losses.append(final_reconst_loss)
                initial_losses.append(reconst_loss)

            np.save(f'results/final_mse_loss_{save_method}.npy', final_losses)
            np.save(f'results/initial_mse_loss_{save_method}.npy', initial_losses)










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
    x_dim = training_data.data.shape[1]**2
    learning_rate = 1e-3
    avgnum = 16
    T = 300
    hidden_dim = 256
    encoder_dim = [-1, 1, 16, 16]
    encoder_method = 'VAE_256'
    method = f'Diffusion'
    model_method = 'Diffusion'
    save_method = f'Diffusion_A_forsample_batchsize1'
    batchnum = 1
    noise = True



    # Load model
    model = SimpleUnet().to(device)
    encoder = VAE(x_dim, hidden_dim).to(device)
    encoder.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + encoder_method + '.pth'))
    for param in encoder.parameters():
        param.requires_grad = False
    try:
        model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + model_method + '.pth'))
    except:
        print('wrong model')
        model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + 'Diffusion' + '.pth'))

    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))



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




    test_A_forsample(batchnum, save_method)
    # test_A_forsample_SNR(batch_size, save_method)

    # test_A_forsample_oneimg(save_method)
    # test_A_forsample_singular(batch_size, save_method)





