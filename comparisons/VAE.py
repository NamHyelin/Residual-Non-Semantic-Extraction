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
from VAE_model import VAE, train_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train_1(method, input_dim, training_loader, validation_loader, num_epochs, lr, batch_size, classifier):
    mse_losses, sem_losses = np.zeros((14)), np.zeros((14))
    for d in np.arange(2, 28, 2):
        c = int((256 + 28 * d)/2)
        print('Hidden dim: ', c)
        model = VAE(input_dim, c).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for e in range(num_epochs):
            for batch_idx_tr, (x, _) in enumerate(training_loader):
                model.train()
                model.train()
                x = x.view(-1, x_dim)
                x = x.to(device)

                optimizer.zero_grad()
                mu, logvar, z, recon_x = model(x)

                B = 0.5
                reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x, recon_x=recon_x, mu_1=mu, logvar_1=logvar)
                loss = (1 - B) * reconst_loss + B * regular_loss

                loss.backward()
                optimizer.step()
            print(f'[{e}/{num_epochs}] reconst loss: {reconst_loss}, regular loss: {regular_loss}')

        mse_loss, sem_loss = 0.0, 0.0
        for batch_idx_ts, (x, y) in enumerate(validation_loader):
            x = x.view(-1, x_dim)
            x, y = x.to(device), y.to(device)
            mu, logvar, z, recon_x = model(x)
            mse_loss += nn.MSELoss()(recon_x, x).item() / batch_size
            sem_loss += (y.size(0) - (classifier(recon_x.view(-1,1,28,28)).argmax(dim=1) == y).int().sum().item()) / batch_size

        mse_losses[int(d / 2) - 1] = mse_loss / batch_idx_ts
        sem_losses[int(d / 2) - 1] = sem_loss / batch_idx_ts

        np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{method}.npy', mse_losses)
        np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{method}.npy', sem_losses)




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
    hidden_dim = 128
    lr = 1e-3
    method= 'VAE_1'

    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))

    train_1(method, x_dim, training_loader, validation_loader, num_epochs, lr, batch_size, classifier)
