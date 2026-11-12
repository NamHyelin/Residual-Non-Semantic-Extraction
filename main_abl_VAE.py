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
from torch.utils.data import DataLoader, Subset
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
from SNR_VAE import add_noise_qam, add_noise_gau
from VAE_model import VAE
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train_avggen(method, avgnum, noise):
    for batch_idx_tr, (x, _) in enumerate(training_loader):
        model.train()
        x = x.view(-1, x_dim)
        x = x.to(device)


        for i in range(batch_size):
            mu, logvar = model.encode(x[i].unsqueeze(0))
            for g in range(avgnum):
                if noise == True:
                    ber_index = random.randint(1, len(snr)+1)
                    mu, logvar = add_noise(mu, ber_index-1), add_noise(logvar, ber_index - 1)
                z = model.reparameterize(mu, logvar)
                recon_x = model.decode(z)
                if g==0:
                    x_, recon_x_ = x[i].unsqueeze(0), recon_x
                else:
                    x_, recon_x_ = torch.cat((x_,x[i].unsqueeze(0)), 0), torch.cat((recon_x_, recon_x), 0)

            #TODO calculate A
            print('saving A for img generation ', i)
            for d in np.arange(2, 28, 2):
                x_, recon_x_ = x_.cpu(), recon_x_.cpu()
                if '_A' in method:
                    # smalleigen
                    A = calcul_A_MNIST(x_, recon_x_, d, False).cpu()
                else:
                    A = calcul_A_MNIST(x_, recon_x_, d, True).cpu()  # (d, 28)
                try:
                    torch.save(A, f'As/{method}/A_d{d}_imgidx{i}.pt')
                except:
                    os.makedirs(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}')
                    torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_imgidx{i}.pt')

        if batch_idx_tr ==0 :
            sys.exit(0)





def test_avggen(method, save_method, avgnum, imgnum, noise):
    with torch.no_grad():
        semantic_mse_loss, final_mse_loss = 0.0, 0.0
        sem_losses, d_losses=np.zeros((14)), np.zeros((14))
        for batch_idx, (x_, y) in enumerate(validation_loader):

            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x, y = x.to(device), y.to(device)

            mu, logvar = model.encode(x.unsqueeze(0))
            if noise == True:
                ber_index = random.randint(1, len(snr) + 1)
                mu, logvar = add_noise(mu, ber_index - 1), add_noise(logvar, ber_index - 1)
            z = model.reparameterize(mu, logvar)
            recon_x = model.decode(z)

            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
            class_error = (y.size(0) - (classifier(recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)


            #TODO plot under-expessed term
            d_loss, sem_loss = [], []
            d_loss.append(reconst_loss/batch_size)
            sem_loss.append(class_error / batch_size)
            for d in np.arange(2, 28, 2):
                A = torch.zeros((d,training_data.data.shape[1])).to(device)
                for c in range(imgnum):
                    A += torch.load(f'As/{method}/A_d{d}_imgidx{c}.pt').to(device)
                A /= c


                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()
                final_class_error = (y.size(0) - (classifier(final_recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)
                d_loss.append(final_reconst_loss/batch_size)
                sem_loss.append(final_class_error / batch_size)
            d_losses += d_loss
            sem_losses += sem_loss


        np.save(f'results/final_mse_loss_{save_method}_avgnum{avgnum}_imgnum{imgnum}.npy', d_losses / batch_idx)
        np.save(f'results/final_classify_error_{save_method}_avgnum{avgnum}_imgnum{imgnum}.npy', sem_losses / batch_idx)

        print('final losses wrt d: ', d_losses/batch_idx)
        plt.subplot(1,2,1)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')

        plt.subplot(1, 2, 2)
        plt.plot(sem_losses / batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('Classification loss')
        plt.title('Reconstruction error wrt d')
        plt.show()
        plt.show()






def train_avgall(method, avgnum, noise):
    for batch_idx_tr, (x, _) in enumerate(training_loader):
        model.train()
        x = x.view(-1, x_dim)
        x = x.to(device)


        for i in range(batch_size):
            mu, logvar = model.encode(x[i].unsqueeze(0))
            for g in range(avgnum):
                if noise == True:
                    ber_index = random.randint(1, len(snr) + 1)
                    mu, logvar = add_noise(mu, ber_index - 1), add_noise(logvar, ber_index - 1)
                z = model.reparameterize(mu, logvar)
                recon_x = model.decode(z)
                if g==0:
                    x_, recon_x_ = x[i].unsqueeze(0), recon_x
                else:
                    x_, recon_x_ = torch.cat((x_,x[i].unsqueeze(0)), 0), torch.cat((recon_x_, recon_x), 0)

        #TODO calculate A
        print('saving A for img generation ', i)
        for d in np.arange(2, 28, 2):
            x_, recon_x_ = x_.cpu(), recon_x_.cpu()
            if '_A' in method:
                # smalleigen
                A = calcul_A_MNIST(x_, recon_x_, d, False).cpu()
            else:
                A = calcul_A_MNIST(x_, recon_x_, d, True).cpu()  # (d, 28)
            try:
                torch.save(A, f'As/{method}/A_d{d}_bidx0.pt')
            except:
                os.makedirs(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}')
                torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx0.pt')

        if batch_idx_tr ==0 :
            sys.exit(0)





def test_avgall(method, save_method, avgnum, noise):
    with torch.no_grad():
        semantic_mse_loss, final_mse_loss = 0.0, 0.0
        sem_losses, d_losses=np.zeros((14)), np.zeros((14))
        for batch_idx, (x_, y) in enumerate(validation_loader):

            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x, y = x.to(device), y.to(device)

            mu, logvar = model.encode(x.unsqueeze(0))
            if noise == True:
                ber_index = random.randint(1, len(snr) + 1)
                mu, logvar = add_noise(mu, ber_index - 1), add_noise(logvar, ber_index - 1)
            z = model.reparameterize(mu, logvar)
            recon_x = model.decode(z)

            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
            class_error = (y.size(0) - (classifier(recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)


            #TODO plot under-expessed term
            d_loss, sem_loss = [], []
            d_loss.append(reconst_loss/batch_size)
            sem_loss.append(class_error / batch_size)
            for d in np.arange(2, 28, 2):
                A = torch.load(f'As/{method}/A_d{d}_bidx0.pt').to(device)


                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()
                final_class_error = (y.size(0) - (classifier(final_recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)
                d_loss.append(final_reconst_loss/batch_size)
                sem_loss.append(final_class_error / batch_size)
            d_losses += d_loss
            sem_losses += sem_loss


        np.save(f'results/final_mse_loss_{save_method}_avgnum{avgnum}_bidx0.npy', d_losses / batch_idx)
        np.save(f'results/final_classify_error_{save_method}_avgnum{avgnum}_bidx0.npy', sem_losses / batch_idx)

        print('final losses wrt d: ', d_losses/batch_idx)
        plt.subplot(1,2,1)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')

        plt.subplot(1, 2, 2)
        plt.plot(sem_losses / batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('Classification loss')
        plt.title('Reconstruction error wrt d')
        plt.show()
        plt.show()








def test_A_forsample(batchsize, save_method):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses=np.zeros((14))
        for batch_idx, (x_, y) in enumerate(training_loader):
            print(f'batch: [{batch_idx}/{int(1000/batch_size)}]')
            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar, z, recon_x = model(x)
            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()


            #TODO plot under-expessed term
            d_loss = []
            d_loss.append(reconst_loss/batch_size)
            # for d in np.arange(2, 28, 2):
            for d in np.arange(2, 28, 2):
                A = calcul_A_MNIST(x, recon_x, d, True).to(device)
                # A = calcul_A_singular_MNIST(x, recon_x, 28, d, True).to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()
                d_loss.append(final_reconst_loss/batch_size)
            d_losses += d_loss

            if batch_size*batch_idx > 1000:
                break

        if batch_idx==0: batch_idx=1
        if type(batchsize) == str:
            np.save(f'results/final_mse_loss_{save_method}.npy', d_losses/batch_idx)
        else:
            np.save(f'results/final_mse_loss_{save_method}.npy', d_losses / batch_idx)

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
        ranks = np.zeros((len(snr)+1, len(np.arange(2, 28, 2))+1))
        for batch_idx, (x_, y) in enumerate(training_loader):
            print('Batch: ', batch_idx)

            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar = model.encode(x)

            for ber_index in np.arange(1, len(snr)+1):
                mu, logvar = add_noise_gau(mu, ber_index-1), add_noise_gau(logvar, ber_index-1)

                z = model.reparameterize(mu, logvar)
                recon_x = model.decode(z)

                reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
                d_losses[ber_index - 1, 0] += reconst_loss / batchsize


                #TODO plot under-expessed term
                for d in np.arange(2, 28, 2):
                    A = calcul_A_MNIST(x, recon_x, d, True).to(device)
                    A = add_noise_gau(A, ber_index - 1)

                    Ax = torch.matmul(A, x.view(-1,28,28))

                    A_ = torch.linalg.pinv(A)
                    final_recon_x = torch.matmul(A_, Ax) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))


                    final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()

                    d_losses[ber_index - 1, int(d / 2)] += final_reconst_loss / batchsize
                    ranks[ber_index - 1, int(d / 2)] += calcul_A_eigenvalues(x, recon_x)

            if batch_size * batch_idx > 10000:
                break

        np.save(f'results/final_mse_loss_{save_method}.npy', d_losses / batch_idx)
        np.save(f'results/final_ranks_{save_method}.npy', ranks / batch_idx)

        # print('original loss: ', origin_mse_loss/batch_idx, '\n')
        # print('final losses wrt d: ', d_losses/batch_idx)







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
            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar, z, recon_x = model(x)
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








def test_A_forsample_mixMnist(save_method, training_data, classes):
    indices = [i for i, target in enumerate(training_data.targets) if target in classes]
    training_data = Subset(training_data, indices)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    with torch.no_grad():
        d_losses = np.zeros((14))
        for batch_idx, (x_, y) in enumerate(training_loader):
            print(f'batch: [{batch_idx}/{int(1000 / batch_size)}]')
            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar, z, recon_x = model(x)
            reconst_loss = nn.MSELoss()(recon_x.view(-1, 28, 28), x.view(-1, 28, 28)).item()

            # TODO plot under-expessed term
            d_loss = []
            d_loss.append(reconst_loss / batch_size)
            # for d in np.arange(2, 28, 2):
            for d in np.arange(2, 28, 2):
                A = calcul_A_MNIST(x, recon_x, d, True).to(device)
                # A = calcul_A_singular_MNIST(x, recon_x, 28, d, True).to(device)

                Ax = torch.matmul(A, x.view(-1, 28, 28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1, 28, 28)).item()
                d_loss.append(final_reconst_loss / batch_size)
            d_losses += d_loss

            if batch_size * batch_idx > 1000:
                break

        d_losses /= batch_idx

        np.save(f'results/final_mse_loss_{save_method}.npy', d_losses)






















def test_A_forsample_oneimg(save_method):
    with torch.no_grad():
        final_losses = []
        initial_losses =[]
        for batch_idx, (x_, y) in enumerate(training_loader):
            for imgidx in range(batch_size):
                x = x_[imgidx].unsqueeze(0).view(1, x_dim)

                x = x.view(-1, x_dim)
                x = x.to(device)

                mu, logvar, z, recon_x = model(x)
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
    x_dim = training_data.data.shape[1]**2
    hidden_dim = 128
    learning_rate = 1e-3
    avgnum = 16
    method = f'VAE'
    model_method = 'VAE'
    save_method = f'VAE_mixMNIST_class0123456'  #f'VAE_mixMNIST_d4_class0'
    imgnum = 50
    noise = True



    # Load model
    model = VAE(x_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/'+ model_method +'.pth'))

    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))



    # train_avggen(method, avgnum, noise) #method = f'VAE_avggen_avgnum{avgnum}'
    # test_avggen(method, save_method, avgnum, imgnum, noise)  #method = f'VAE_avggen_avgnum{avgnum}' save_method = f'VAE_avggen'

    # train_avgall(method, avgnum) #method = f'VAE_avgall_avgnum{avgnum}'
    # test_avgall(method, save_method, avgnum)  #method = f'VAE_avgall_avgnum{avgnum}' save_method = f'VAE_avggen'

    test_A_forsample(batch_size, save_method)
    # test_A_forsample_SNR(batch_size, save_method)

    # test_A_forsample_oneimg(save_method)
    # test_A_forsample_singular(batch_size, save_method)

    test_A_forsample_mixMnist(save_method=save_method, training_data=training_data, classes=[0,1,2,3,4,5,6])





