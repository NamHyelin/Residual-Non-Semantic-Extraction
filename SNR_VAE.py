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
from QAM import Ps_qam, snr



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train_loss(batch_size, x, recon_x, mu_1, logvar_1, mu_2 = None, logvar_2 = None):
    if mu_2 == None and logvar_2==None:
        mu_2 = torch.zeros_like(mu_1)
        logvar_2 = torch.log(torch.ones_like(logvar_1))
    reconst_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size # nn.MSELoss()(recon_x, x) # nn.BCEWithLogitsLoss()(recon_x, x) / batch_size
    regular_loss = torch.mean(0.5 * ((1/(logvar_2.exp()))*((mu_1-mu_2).pow(2)) + (logvar_1.exp())/(logvar_2.exp())  - 1 - (logvar_1-logvar_2))) #sum 막해도되는지
    return reconst_loss, regular_loss




def add_noise_qam(data, snr_idx):
    if snr_idx<len(Ps_qam):
        ber = Ps_qam[snr_idx]
        num_dim = data.numel()
        erase_indices = np.random.choice(num_dim, size=int(num_dim*ber), replace=False)
        data_flat = data.contiguous().view(-1)
        data_flat[erase_indices] = 0
        data_return = data_flat.view(data.size())
    else:
        data_return = data
    return data_return




def add_noise_gau(data, snr_idx):
    snrdb = snr[snr_idx]
    fading= np.random.normal(0.8, 1)
    with torch.no_grad():
        noise_power = torch.mean(data**2) / (10**(snrdb/10))
        try:
            gauss_noise = torch.normal(0.0, noise_power.sqrt().item(), data.size())/4  # gain?
        except:
            print('hi')
            gauss_noise=torch.ones_like(data)
    data_return = data*fading + gauss_noise.to(device)
    return data_return











def main_all_SNR():
    # secondary
    # noise 를 랜덤하게 줌
    # train : train을 할건지
    # train noise : train 할 떄 noise 를 낄건지
    # adapt noise : A를 계산할때 noise 를 낄건지

    #ebno = np.logspace(0.1, 1.0, 10)  # 1.2, 12
    #snr = 10 * np.log10(ebno)

    reconst_loss_train, regular_loss_train, Bs = [], [], []
    reconst_loss_test, regular_loss_test, semantic_loss_test, num_clusters_test = [], [], [], []
    for e in range(num_epochs):
        reconst_loss_tr, regular_loss_tr = 0.0, 0.0
        reconst_loss_ts, regular_loss_ts, semantic_loss_ts, num_clusters_ts = 0.0, 0.0, 0.0, 0.0

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            print(f'Batch train: [{batch_idx_tr}] / 600')
            model.train()
            x = x.view(-1, x_dim)
            x = x.to(device)

            if train==True:
                optimizer.zero_grad()
                mu, logvar = model.encode(x)

                #TODO block bits in mu, logvar
                if train_noise == True:
                    ber_index = random.randint(1, len(snr)+1)
                    mu, logvar = add_noise_qam(mu, ber_index-1), add_noise_qam(logvar, ber_index - 1)

                z = model.reparameterize(mu, logvar)
                recon_x = model.decode(z)


                B = 0.5
                reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x, recon_x=recon_x, mu_1=mu, logvar_1=logvar)
                loss = (1 - B) * reconst_loss + B * regular_loss


                reconst_loss_tr += reconst_loss.item()
                regular_loss_tr += regular_loss.item()

                loss.backward()
                optimizer.step()
            else:
                model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + model_method + '.pth'))



            #TODO calculate A
            if e == num_epochs-1 :
                print('saving A for batch: ', batch_idx_tr)
                if adapt_noise==True:
                    mu, logvar = model.encode(x)
                    ber_index = random.randint(1, len(snr) + 1)
                    mu, logvar = add_noise_qam(mu, ber_index - 1), add_noise_qam(logvar, ber_index - 1)
                    z = model.reparameterize(mu, logvar)
                    recon_x = model.decode(z)

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
                    print(f'Test : [{batch_idx_ts}/100]')
                    x_ = x.view(batch_size, x_dim)
                    x, x_, label = x.to(device), x_.to(device), label.to(device)
                    mu, logvar = model.encode(x_)

                    # TODO noise add to mu, logvar
                    ber_index = random.randint(1, len(snr) + 1)
                    mu, logvar = add_noise(mu, ber_index - 1), add_noise(logvar, ber_index - 1)


                    z = model.reparameterize(mu, logvar)
                    recon_x = model.decode(z)
                    reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x_, recon_x=recon_x, mu_1=mu, logvar_1=logvar)

                    classifier_loss = nn.CrossEntropyLoss()(classifier(recon_x.view(batch_size, 1, 28, 28)), label)

                    reconst_loss_ts += reconst_loss.item()
                    regular_loss_ts += regular_loss.item()
                    semantic_loss_ts += classifier_loss.item()


        # reconst_loss_train.append(reconst_loss_tr / batch_idx_tr)
        # regular_loss_train.append(regular_loss_tr / batch_idx_tr)
        # reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
        # regular_loss_test.append(regular_loss_ts / batch_idx_ts)
        # semantic_loss_test.append(semantic_loss_ts / batch_idx_ts)
        #
        # print(f'[{e} / {num_epochs}] reconstruction loss: {reconst_loss_test[-1]}, regularization loss: {regular_loss_test[-1]}, semantic loss: {semantic_loss_test[-1]}')



        # np.save('results/reconst_loss_train_'+ method + '.npy', reconst_loss_train)
        # np.save('results/regular_loss_train_' + method + '.npy', regular_loss_train)
        # np.save('results/reconst_loss_test_' + method + '.npy', reconst_loss_test)
        # np.save('results/regular_loss_test_' + method + '.npy', regular_loss_test)
        # np.save('results/semantic_loss_test_' + method + '.npy', semantic_loss_test)
        # np.save('results/B_' + method + '.npy', Bs)
        if train==True:
            torch.save(model.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/'+model_method+'.pth')










def main_one_SNR(ber_index):
    # instantan, second order
    # 특정 snr에서만
    # adapt noise 따로 나누지 않음. train때 있으면 adapt noise 도 똑같이 있는거임

    reconst_loss_train, regular_loss_train, Bs = [], [], []
    reconst_loss_test, regular_loss_test, semantic_loss_test, num_clusters_test = [], [], [], []
    for e in range(num_epochs):
        reconst_loss_tr, regular_loss_tr = 0.0, 0.0
        reconst_loss_ts, regular_loss_ts, semantic_loss_ts, num_clusters_ts = 0.0, 0.0, 0.0, 0.0
        ranks = []

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            # print(f'Batch train: [{batch_idx_tr}] / 600')
            model.train()
            x = x.view(-1, x_dim)
            x = x.to(device)

            if train==True:
                optimizer.zero_grad()
                mu, logvar = model.encode(x)


                #TODO block bits in mu, logvar
                if train_noise == True:
                    mu, logvar = add_noise_gau(mu, ber_index), add_noise_gau(logvar, ber_index)

                z = model.reparameterize(mu, logvar)
                recon_x = model.decode(z)


                B = 0.5
                reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x, recon_x=recon_x, mu_1=mu, logvar_1=logvar)
                loss = (1 - B) * reconst_loss + B * regular_loss

                if HFR_desnois==True:
                    d = np.random.choice(np.arange(2,28,2), 1)[0] #d는 랜덤으로 골라
                    x_, recon_x_ = x.cpu(), recon_x.cpu()
                    A = calcul_A_MNIST(x, recon_x, d, True).to(device)
                    Ax = torch.matmul(A, x.view(-1, 28, 28))
                    Ax = add_noise_gau(Ax, ber_index - 1)
                    A = add_noise_gau(A, ber_index - 1)
                    A_ = torch.linalg.pinv(A)
                    final_recon_x = torch.matmul(A_, Ax) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                    #clamp 를 해야됨
                    reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x, recon_x=torch.clamp(final_recon_x, min=0, max=1).view(-1,x_dim), mu_1=mu, logvar_1=logvar)
                    loss = (1 - B) * reconst_loss + B * regular_loss

                reconst_loss_tr += reconst_loss.item()
                regular_loss_tr += regular_loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + model_method + '.pth'))



            #TODO calculate A
            if e == num_epochs-1 :
                print('saving A for batch: ', batch_idx_tr)
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
                    # print(f'Test : [{batch_idx_ts}/100]')
                    x_ = x.view(batch_size, x_dim)
                    x, x_, label = x.to(device), x_.to(device), label.to(device)
                    mu, logvar = model.encode(x_)

                    # TODO noise add to mu, logvar
                    mu, logvar = add_noise_qam(mu, ber_index), add_noise_qam(logvar, ber_index)


                    z = model.reparameterize(mu, logvar)
                    recon_x = model.decode(z)
                    reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x_, recon_x=recon_x, mu_1=mu, logvar_1=logvar)

                    classifier_loss = nn.CrossEntropyLoss()(classifier(recon_x.view(batch_size, 1, 28, 28)), label)

                    reconst_loss_ts += reconst_loss.item()
                    regular_loss_ts += regular_loss.item()
                    semantic_loss_ts += classifier_loss.item()

        if train == True:
            reconst_loss_train.append(reconst_loss_tr / batch_idx_tr)
            regular_loss_train.append(regular_loss_tr / batch_idx_tr)
            reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
            regular_loss_test.append(regular_loss_ts / batch_idx_ts)
            semantic_loss_test.append(semantic_loss_ts / batch_idx_ts)

            print(f'[{e} / {num_epochs}] reconstruction loss: {reconst_loss_test[-1]}, regularization loss: {regular_loss_test[-1]}, semantic loss: {semantic_loss_test[-1]}')



        # np.save('results/reconst_loss_train_'+ method + '.npy', reconst_loss_train)
        # np.save('results/regular_loss_train_' + method + '.npy', regular_loss_train)
        np.save('results/reconst_loss_test_' + method + '.npy', reconst_loss_test)
        np.save('results/regular_loss_test_' + method + '.npy', regular_loss_test)
        # np.save('results/semantic_loss_test_' + method + '.npy', semantic_loss_test)
        # np.save('results/B_' + method + '.npy', Bs)
        if train==True:
            torch.save(model.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/'+model_method+'.pth')




















if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()])) #, transforms.Normalize((0.1307), (0.3081))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()])) #, transforms.Normalize((0.1307), (0.3081))
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
    num_epochs = 50
    x_dim = training_data.data.shape[1]**2
    hidden_dim = 240 #128
    learning_rate = 1e-3

    ber_index =10
    mysnr = int(snr[ber_index] ) #snr= [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    method= f'VAE_240_denoising_awgn_SNR{mysnr}'
    train = True
    model_method = f'VAE_240_denoising_awgn_SNR{mysnr}'
    train_noise= True
    adapt_noise= True
    HFR_desnois= False



    # Load model
    model = VAE(x_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))



    # main_all_SNR()
    main_one_SNR(ber_index)

