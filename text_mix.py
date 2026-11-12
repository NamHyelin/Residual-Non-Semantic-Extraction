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
from classifier import CNN
from unexpressed import calcul_A_MNIST
from VAE_model import VAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def test_mix(classifier, model_enc, model_dec, A_method, bidx, method, dataloader, batch_size, x_dim, dd):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses=np.zeros((14))
        cls_losses = np.zeros((14))

        for batch_idx, (x_, y) in enumerate(dataloader):

            x = x_.view(batch_size, x_dim)
            x = x.view(-1, x_dim)
            x, y = x.to(device), y.to(device)

            z = model_enc(x)
            recon_x = model_dec(z)

            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
            class_error = (y.size(0) - (classifier(recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)

            #TODO plot under-expessed term
            d_loss = []
            cls_loss = []
            d_loss.append(reconst_loss/batch_size)
            cls_loss.append(class_error / batch_size)

            for d in np.arange(2, 28, 2):
                if type(bidx)==str:
                    A = torch.zeros((d,x_.shape[2])).to(device)
                    for c in range(int(bidx)):
                        A += torch.load(f'As/{A_method}/A_d{d}_bidx{c}.pt').to(device)
                    A /= c
                else:
                    A = torch.load(f'As/{A_method}/A_d{d}_bidx{bidx}.pt').to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                final_recon_x = torch.matmul(torch.matmul(A_,A), x.view(-1,28,28)) + torch.matmul((torch.eye(28).to(device)-torch.matmul(A_,A)) ,recon_x.view(-1,28,28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()
                class_error = (y.size(0) - (classifier(final_recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)

                d_loss.append(final_reconst_loss/batch_size)
                cls_loss.append(class_error/batch_size)

                if d==dd:
                    if batch_idx==0:
                        for j in range(10):
                            plt.imsave(f'C:/TransUnrecon/VAE_MNIST_2/results/samples/mix/{method}_Ax_{j}.jpg', Ax[j].detach().cpu().numpy(), cmap='gray')
                            plt.imsave(f'C:/TransUnrecon/VAE_MNIST_2/results/samples/mix/{method}_x_{j}.jpg', x[j].view(28, 28).detach().cpu().numpy(), cmap='gray')
                            plt.imsave(f'C:/TransUnrecon/VAE_MNIST_2/results/samples/mix/{method}_reconx_{j}.jpg', recon_x[j].view(28, 28).detach().cpu().numpy(), cmap='gray')
                            plt.imsave(f'C:/TransUnrecon/VAE_MNIST_2/results/samples/mix/{method}_finalreconx_{j}.jpg', final_recon_x[j].view(28, 28).detach().cpu().numpy(), cmap='gray')
            d_losses += d_loss
            cls_losses += cls_loss


        # if type(bidx) == str:
        #     np.save(f'results/final_mse_loss_{method}_bidx_avg'+bidx+'.npy', d_losses/batch_idx)
        # else:
        #     np.save(f'results/final_mse_loss_{method}_bidx_{bidx}.npy', d_losses / batch_idx)
        #
        # if type(bidx) == str:
        #     np.save(f'results/final_classify_error_{method}_bidx_avg'+bidx+'.npy', cls_losses/batch_idx)
        # else:
        #     np.save(f'results/final_classify_error_{method}_bidx_{bidx}.npy', cls_losses / batch_idx)

        print('original loss: ', origin_mse_loss/batch_idx, '\n')
        print('final losses wrt d: ', d_losses/batch_idx)
        plt.subplot(1,2,1)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')

        plt.subplot(1, 2, 2)
        plt.plot(cls_losses / batch_idx, label='')
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('Classification error')
        plt.title('Semantic error4 wrt d')
        plt.show()









if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
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
    dd=2
    method= f'samples_d{dd}'
    bidx = 0



    # Load model
    enc_method = 'VAE'
    dec_method = 'VAE'
    A_method = 'VAE'
    classifier_method = 'classifier_mnist'  # encoder = dataset = classifier



    enc_model = VAE(x_dim, hidden_dim).to(device)
    dec_model = VAE(x_dim, hidden_dim).to(device)
    enc_model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/'+enc_method+'.pth'))
    dec_model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/'+dec_method+'.pth'))
    def model_enc(x):
        mu, logvar = enc_model.encode(x)
        return enc_model.reparameterize(mu, logvar)
    def model_dec(x):
        return dec_model.decode(x)


    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(f'C:/TransUnrecon/VAE_MNIST_2/{classifier_method}.pth'))


    test_mix(classifier, model_enc, model_dec, A_method, bidx, method, training_loader, batch_size, x_dim, dd)
