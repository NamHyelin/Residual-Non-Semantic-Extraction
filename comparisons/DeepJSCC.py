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
from BER import Ax_error
from lpips_score import lpips, pfw, LPIPS, FID, KID


# https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission/blob/master/Keras_Implementation/AutoencoderModel.py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class NormalizationNoise(nn.Module):
    def __init__(self, snr_db_def=20, P_def=1):
        super(NormalizationNoise, self).__init__()
        self.snr_db = nn.Parameter(torch.tensor(snr_db_def, dtype=torch.float32), requires_grad=False)
        self.P = nn.Parameter(torch.tensor(P_def, dtype=torch.float32), requires_grad=False)

    def forward(self, z_tilda):
        # Convert to complex and compute the necessary parameters
        z_tilda = torch.complex(z_tilda, torch.zeros_like(z_tilda)) + 1j
        k = np.prod(z_tilda.size()[1:])
        z_conjugateT = torch.conj(z_tilda.permute(0, 1, 3, 2))
        sqrt1 = torch.sqrt(k * self.P)
        sqrt2 = torch.sqrt(torch.mul(z_conjugateT, z_tilda).sum(dim=[1, 2, 3], keepdim=True))

        # Normalization and power constraint
        z = sqrt1 * z_tilda / sqrt2
        pwr = torch.mean(torch.abs(z) ** 2)
        Z = z / torch.sqrt(pwr / self.P) if pwr > self.P else z

        # AWGN
        snr = 10 ** (self.snr_db / 10.0)
        sig_pwr = torch.mean(torch.abs(Z) ** 2)
        noise_pwr = sig_pwr / snr
        noise_sigma = torch.sqrt(noise_pwr / 2)

        noise = torch.randn_like(torch.view_as_real(Z)) * noise_sigma
        z_cap = torch.view_as_complex(noise) + Z

        return torch.view_as_real(z_cap)[:,:,:,:,0]


class AutoEncoder(nn.Module):
    def __init__(self, c):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, c, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(c),
            nn.PReLU(),

        )

        #(batchsize, c, 4, 4)

        self.normalization_noise = NormalizationNoise()

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.normalization_noise(x)
        x = self.decoder(x)
        crop_height, crop_width = 14, 14
        x = x[:, :, crop_height:-crop_height, crop_width:-crop_width]
        return x







def train_1(training_loader, validation_loader, num_epochs, lr, snr, classifier):
    mse_losses, sem_losses = np.zeros((14)), np.zeros((14))
    for d in np.arange(2, 28, 2):
        c = int((256 + 28 * d) / 16)
        print('Hidden dim: ', c)
        model = AutoEncoder(c).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.normalization_noise.snr_db.data = torch.tensor(snr, dtype=torch.float32)

        for e in range(num_epochs):
            for batch_idx_tr, (x, _) in enumerate(training_loader):
                model.train()
                optimizer.zero_grad()
                x = x.to(device)
                recon_x = model(x)
                loss = nn.MSELoss()(recon_x, x)
                loss.backward()
                optimizer.step()
            if e == int(num_epochs / 2):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr / 10
            print(f'd{d}, [{e}/{num_epochs}] loss: {loss.item()}')

        mse_loss, sem_loss = 0.0, 0.0
        for batch_idx_ts, (x, y) in enumerate(validation_loader):
            x, y = x.to(device), y.to(device)
            recon_x = model(x)
            mse_loss += nn.MSELoss()(recon_x, x).item() / batch_size
            sem_loss += (y.size(0) - (classifier(recon_x).argmax(dim=1) == y).int().sum().item()) / batch_size

        mse_losses[int(d / 2) - 1] = mse_loss / batch_idx_ts
        sem_losses[int(d / 2) - 1] = sem_loss / batch_idx_ts

        np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_DeepJSCC_1.npy', mse_losses)
        np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_DeepJSCC_1.npy', sem_losses)




def train_2(c, bidx, training_loader, validation_loader, num_epochs, lr, snr, classifier):
    model = AutoEncoder(c).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.normalization_noise.snr_db.data = torch.tensor(snr, dtype=torch.float32)
    method = 'DeepJSCC_2'

    reconst_loss_test, semantic_loss_test = [], []
    for e in range(num_epochs):
        reconst_loss_ts, semantic_loss_ts = 0.0, 0.0

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            model.train()
            optimizer.zero_grad()
            x = x.to(device)
            recon_x = model(x)
            loss = nn.MSELoss()(recon_x, x)
            loss.backward()
            optimizer.step()

            # TODO calculate A
            if e == num_epochs - 1:
                print('saving A for batch ', batch_idx_tr)
                for d in np.arange(2, 28, 2):
                    x, recon_x = x.cpu(), recon_x.cpu()
                    A = calcul_A_MNIST(x, recon_x, d, True).cpu()  # (d, 28)
                    try:
                        torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')
                    except:
                        os.makedirs(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}')
                        torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')
        if e == int(num_epochs / 2):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / 10

        with torch.no_grad():
            for batch_idx_ts, (x, label) in enumerate(validation_loader):
                x, label = x.to(device), label.to(device)
                recon_x = model(x)

                reconst_loss = nn.MSELoss()(recon_x, x)
                classifier_loss = nn.CrossEntropyLoss()(classifier(recon_x), label)

                reconst_loss_ts += reconst_loss.item() / batch_size
                semantic_loss_ts += classifier_loss.item() / batch_size

        reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
        semantic_loss_test.append(semantic_loss_ts / batch_idx_ts)

        print(f'[{e} / {num_epochs}] reconstruction loss: {reconst_loss_test[-1]}, semantic loss: {semantic_loss_test[-1]}')

        np.save('C:/TransUnrecon/VAE_MNIST_2/results/comparisons/reconst_loss_test_' + method + '.npy', reconst_loss_test)
        np.save('C:/TransUnrecon/VAE_MNIST_2/results/comparisons/semantic_loss_test_' + method + '.npy', semantic_loss_test)
        torch.save(model.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/' + method + '.pth')



    print('\ntest with A')
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses, c_errors = np.zeros((14)), np.zeros((14))
        for batch_idx, (x, y) in enumerate(validation_loader):
            x = x.to(device)

            recon_x = model(x)
            reconst_loss = nn.MSELoss()(recon_x, x).item()
            classify_error = nn.CrossEntropyLoss()(classifier(recon_x), label).item()

            #TODO plot under-expessed term
            d_loss, c_error = [], []
            d_loss.append(reconst_loss/batch_size)
            c_error.append(classify_error / batch_size)
            for d in np.arange(2, 28, 2):
                if type(bidx)==str:
                    A = torch.zeros((d,training_data.data.shape[1])).to(device)
                    for j in range(int(bidx)):
                        A += torch.load(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{j}.pt').to(device)
                    A /= j
                else:
                    A = torch.load(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{bidx}.pt').to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                final_recon_x = torch.matmul(torch.matmul(A_,A), x) + torch.matmul((torch.eye(28).to(device)-torch.matmul(A_,A)) ,recon_x)

                final_reconst_loss = nn.MSELoss()(final_recon_x, x).item()
                final_classify_error = nn.CrossEntropyLoss()(classifier(final_recon_x), label).item()
                d_loss.append(final_reconst_loss/batch_size)
                c_error.append(final_classify_error/batch_size)
            d_losses += d_loss
            c_errors += c_error


        if type(bidx) == str:
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{method}_bidx_avg'+bidx+'.npy', d_losses/batch_idx)
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{method}_bidx_avg' + bidx + '.npy', d_losses / batch_idx)
        else:
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{method}_bidx_{bidx}.npy', d_losses / batch_idx)
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{method}_bidx_{bidx}.npy', d_losses / batch_idx)
        print('original loss: ', origin_mse_loss/batch_idx, '\n')
        print('final losses wrt d: ', d_losses/batch_idx)
        plt.subplot(1,2,1)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')
        plt.subplot(1, 2, 2)
        plt.plot(c_errors / batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('Classify error')
        plt.title('Semantic error wrt d')
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
    x_dim = training_data.data.shape[1] ** 2
    num_epochs= 50
    lr = 0.0001
    snr = 10
    c = 16
    bidx = 0


    # Load model
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))

    train_1(training_loader, validation_loader, num_epochs, lr, snr, classifier)
    # train_2(c, bidx, training_loader, validation_loader, num_epochs, lr, snr, classifier)






