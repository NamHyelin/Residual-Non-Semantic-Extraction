from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import savgol_filter
from six.moves import xrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from adabelief_pytorch import AdaBelief
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import time
from unexpressed import calcul_A_MNIST
import os
'''
공부하기 좋은
https://ws-choi.github.io/blog-kor/seminar/tutorial/mnist/pytorch/gan/GAN-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC/

GAN 앞뒤로 encoder/decoder 달고 그게 semantic
Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data
Trust-Worthy Semantic Communications for the Metaverse Relying on Federated Learning
'''



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        # range of -1 to 1
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=64)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(in_features=256, out_features=out_features)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.ReLU()

    def forward(self, x):
        # Feed forward
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        tanh_out = self.tanh(x)

        return tanh_out


class Discriminator(nn.Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(in_features=64, out_features=out_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Rehape passed image batch
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # Feed forward
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.dropout(x)
        logit_out = self.fc4(x)

        return logit_out



# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)






if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))

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
    num_epochs = 300
    x_dim = training_data.data.shape[1]**2
    latent_dim = 128*2
    learning_rate = 2e-4
    load = False
    method = f'GAN_{latent_dim}'




    # Initialize model and optimizer
    encoder = Encoder(x_dim, latent_dim).to(device)
    generator = Generator(latent_dim, x_dim).to(device)
    discriminator = Discriminator(x_dim).to(device)

    if load==True:
        encoder.load_state_dict(torch.load(f'C:/TransUnrecon/VAE_MNIST_2/{method}_enc.pth'))
        generator.load_state_dict(torch.load(f'C:/TransUnrecon/VAE_MNIST_2/{method}_gen.pth'))
        discriminator.load_state_dict(torch.load(f'C:/TransUnrecon/VAE_MNIST_2/{method}_dis.pth'))

    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)
    init_params(generator)
    init_params(discriminator)

    optimizer_E = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # optimizer_G = AdaBelief(generator.parameters(), lr=learning_rate,print_change_log=False)
    # optimizer_D = AdaBelief(discriminator.parameters(), lr=learning_rate,print_change_log=False)

    adversarial_loss = nn.BCEWithLogitsLoss()

    reconst_loss_train, reconst_loss_test = [], []

    for epoch in range(num_epochs):
        reconst_loss_tr, reconst_loss_ts = 0.0, 0.0
        d_losses, g_losses = 0.0, 0.0

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            x = x.view(-1, x_dim).to(device)


            #  Train Generator
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            # Sample noise
            # z = torch.randn(batch_size, latent_dim).to(device)
            z = encoder(x)
            recon_x = generator(z)
            g_loss = adversarial_loss(discriminator(recon_x), valid)

            g_loss.backward()
            optimizer_G.step()
            optimizer_E.step()


            #  Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(x), valid)
            fake_loss = adversarial_loss(discriminator(recon_x.detach()), fake)
            d_loss = (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            optimizer_D.zero_grad()
            mseloss = 100*nn.MSELoss()(x, recon_x)
            optimizer_G.step()
            optimizer_E.step()
            optimizer_D.step()

            reconst_loss_tr += g_loss.item()
            d_losses += d_loss.item()
            g_losses += g_loss.item()

            # TODO calculate A
            if epoch == num_epochs - 1:
                print('saving A for batch ', batch_idx_tr)
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
            for batch_idx_ts, (x, _) in enumerate(validation_loader):
                x = x.view(-1, x_dim).to(device)
                z = torch.randn(batch_size, latent_dim).to(device)
                recon_x = generator(z)

                g_loss = nn.MSELoss()(x, recon_x)
                reconst_loss_ts += g_loss.item()

        reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
        reconst_loss_train.append(reconst_loss_tr / batch_idx_tr)

        print(f'[{epoch}/{num_epochs}] Generator loss: {g_losses/batch_idx_tr}, Discriminator loss: {d_losses/batch_idx_tr}')

        np.save(f'results/Bce_loss_train_{method}.npy', reconst_loss_train)
        np.save(f'results/reconst_loss_test_{method}.npy', reconst_loss_test)
        torch.save(encoder.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/{method}_enc.pth')
        torch.save(generator.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/{method}_gen.pth')
        torch.save(discriminator.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/{method}_dis.pth')
