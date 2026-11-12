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
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import time

from classifier import CNN
from main import VAE, get_gauss, train_loss, Tsne_idx
from unexpressed import calcul_A_MNIST
from BER import Ax_error
from lpips_score import lpips, pfw, LPIPS, FID, KID
from VAE_model import VAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class CompressedSensing():
    def __init__(self):
        # https://www.kaggle.com/code/brucelangford/an-introduction-to-compressed-sensing
        from scipy.fft import fft, fftshift, ifft, ifftshift
        import pywt
        self.pywt = pywt

    def compress(self, x, undersample_rate):
        # random undersample. 몇개 bit 에 0을 넣어서 noisy 한 상황 만들기
        n = x.shape[0] * x.shape[1]
        undersampled_x = torch.tensor(x.reshape(-1) * np.random.permutation(np.concatenate(
                (np.ones(int(n * undersample_rate)),
                 np.zeros(int(n-int(n * undersample_rate))))))
                                   ).resize(x.shape[0], x.shape[1])
        return undersampled_x

    def flat_wavelet_transform2(self, x, methodd='bior1.3'):
        """For a 2D image x, take the wavelet """
        coeffs = self.pywt.wavedec2(x, methodd)
        output = coeffs[0].reshape(-1)
        for tups in coeffs[1:]:
            for c in tups:
                output = np.concatenate((output, c.reshape(-1)))
        return output

    def inverse_flat_wavelet_transform2(self, X, shape, methodd='bior1.3'):
        shapes = self.pywt.wavedecn_shapes(shape, methodd)
        nx = shapes[0][0]
        ny = shapes[0][1]
        n = nx * ny
        coeffs = [X[:n].reshape(nx, ny)]
        for i, d in enumerate(shapes[1:]):
            vals = list(d.values())
            nx = vals[0][0]
            ny = vals[0][1]
            coeffs.append((X[n: n + nx * ny].reshape(nx, ny),
                           X[n + nx * ny: n + 2 * nx * ny].reshape(nx, ny),
                           X[n + 2 * nx * ny: n + 3 * nx * ny].reshape(nx, ny)))
            n += 3 * nx * ny
        return self.pywt.waverec2(coeffs, methodd)

    def soft_thresh(self, x, lam):
        if ~(isinstance(x[0], complex)):
            return np.zeros(x.shape) + (x + lam) * (x < -lam) + (x - lam) * (x > lam)
        else:
            return np.zeros(x.shape) + (abs(x) - lam) / abs(x) * x * (abs(x) > lam)


    def plot_wavelet(self, undersampled, methodd='bior1.3'):
        plt.plot(self.flat_wavelet_transform2(undersampled, methodd)[-100:])
        plt.title('A small sample of the wavelet transform')
        plt.show()

    def distance(self, x, y):
        return sum(abs(x.reshape(-1) - y.reshape(-1)))

    def reconstruct(self, x, undersampled_x, steps = 80, methodd = 'sym3', eps = 1e-2, lamda = 100, lamda_decay = 0.995, min_lamda = 1):
        methods = ['haar', 'coif1', 'coif2', 'coif3', 'bior1.1', 'bior1.3', 'bior3.1', 'bior3.3', 'rbio1.1', 'rbio1.3', 'rbio3.1', 'rbio3.3']
        err = []
        n = undersampled_x.shape[0]
        xhat = undersampled_x.numpy().copy()
        for i in range(steps):
            X_hat_old = self.flat_wavelet_transform2(xhat, methodd)
            X_hat = self.soft_thresh(X_hat_old, lamda)
            xhat = self.inverse_flat_wavelet_transform2(X_hat, (n,n), methodd)
            xhat[undersampled_x!=0] = undersampled_x[undersampled_x!=0]

            xhat = xhat.astype(float)
            xhat[xhat < 0] = 0
            xhat[xhat > 255] = 255
            err.append(self.distance(x, xhat))
            lamda *= lamda_decay
            print('step: ', i)

            # if err[-1] < eps:
            #     break
        return lamda, xhat



if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))


    def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(0)
    x_dim = training_data.data.shape[1] ** 2
    batch_size = 100
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True )
    hidden_dim = 128
    wavelet='haar'  #['haar', 'coif1', 'coif2', 'coif3', 'bior1.1', 'bior1.3', 'bior3.1', 'bior3.3', 'rbio1.1', 'rbio1.3', 'rbio3.1', 'rbio3.3']
    step =50
    model_method = 'VAE'
    save_method = 'CompressedSensing_2_diff'


    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))

    model = VAE(x_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + model_method + '.pth'))





#TODO transmitting x
    #
    # with torch.no_grad():
    #     d_mse_losses=np.zeros((10))
    #     d_sem_losses = np.zeros((10))
    #     for batch_idx, (X, y) in enumerate(training_loader):
    #         if batch_idx < 5:
    #             mse_losses, sem_losses = [], []
    #             final_reconst_loss, class_error = 0.0, 0.0
    #             for idx in range(batch_size): #one batch
    #                 x = X[idx].to(device)
    #                 x_ = X[idx].squeeze(0)
    #                 compressed_ratio = (hidden_dim*2) / (x_.size(0)**2)
    #                 undersampled_x= CompressedSensing().compress(x_, compressed_ratio)
    #                 _, final_recon_x = CompressedSensing().reconstruct(x_, undersampled_x, step, wavelet)
    #
    #                 final_recon_x= torch.tensor(final_recon_x).to(device)
    #                 final_reconst_loss += (nn.MSELoss()(final_recon_x.unsqueeze(0), x.view(-1, 28, 28)).item()) / batch_size
    #                 class_error += (y.size(0) - (classifier(final_recon_x.float().view(-1, 1, 28, 28)).argmax(dim=1) == y.to(device)).int().sum().item()) / y.size(0)
    #             mse_losses.append(final_reconst_loss)
    #             sem_losses.append(class_error)
    #
    #             for d in range(2, 20, 2): #one batch  #d=18.85까지 원이미지 보내는거보다 사이즈이득
    #                 final_reconst_loss, class_error = 0.0, 0.0
    #                 for idx in range(batch_size):
    #                     compressed_ratio = ((hidden_dim * 2) + (d * x_.size(0))) / (x_.size(0) ** 2)
    #                     undersampled_x = CompressedSensing().compress(x_, compressed_ratio)
    #                     _, final_recon_x = CompressedSensing().reconstruct(x_, undersampled_x, step, wavelet)
    #
    #                     final_recon_x = torch.tensor(final_recon_x).to(device)
    #                     final_reconst_loss += (nn.MSELoss()(final_recon_x.unsqueeze(0), x.view(-1, 28, 28)).item()) / batch_size
    #                     class_error += (y.size(0) - (classifier(final_recon_x.float().view(-1, 1, 28, 28)).argmax(dim=1) == y.to(device)).int().sum().item()) / y.size(0)
    #                 mse_losses.append(final_reconst_loss)
    #                 sem_losses.append(class_error)
    #
    #         d_mse_losses += mse_losses
    #         d_sem_losses += sem_losses
    #
    #         np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{method}.npy', d_mse_losses / batch_idx)
    #         np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{method}.npy', d_sem_losses / batch_idx)



    #TODO transmitting x- x_hat
    with torch.no_grad():
        d_mse_losses = np.zeros((10))
        d_sem_losses = np.zeros((10))
        for batch_idx, (X, y) in enumerate(training_loader):
            if batch_idx < 5:
                for idx in range(batch_size):
                    x = X[idx].unsqueeze(0).view(1, x_dim)

                    x = x.view(-1, x_dim)
                    x = x.to(device)
                    mu, logvar, z, recon_x = model(x)

                    final_reconst_loss = (nn.MSELoss()(recon_x.view(1, 28, 28), x.view(-1, 28, 28)).item()) / batch_size
                    class_error = (y.size(0) - (classifier(recon_x.float().view(-1, 1, 28, 28)).argmax(dim=1) == y.to(device)).int().sum().item()) / y.size(0)

                    d_mse_losses[0]+=final_reconst_loss
                    d_sem_losses[0]+=class_error

                    for d in range(2, 20, 2):  # one batch  #d=18.85까지 원이미지 보내는거보다 사이즈이득
                        compressed_ratio = d/28
                        undersampled_x = CompressedSensing().compress((recon_x - x).view(28,28).detach().cpu(), compressed_ratio)
                        _, final_recon_x = CompressedSensing().reconstruct((recon_x - x).view(28,28).detach().cpu(), undersampled_x, step, wavelet)

                        final_recon_x = torch.tensor(final_recon_x).to(device) + x.view(28,28)
                        final_reconst_loss = (nn.MSELoss()(final_recon_x.unsqueeze(0), x.view(-1, 28, 28)).item()) / batch_size
                        class_error = (y.size(0) - (classifier(final_recon_x.float().view(-1, 1, 28, 28)).argmax(dim=1) == y.to(device)).int().sum().item()) / y.size(0)

                        d_mse_losses[int(d/2)] += final_reconst_loss
                        d_sem_losses[int(d/2)] += class_error

            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{save_method}.npy', d_mse_losses / batch_idx)
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{save_method}.npy', d_sem_losses / batch_idx)

