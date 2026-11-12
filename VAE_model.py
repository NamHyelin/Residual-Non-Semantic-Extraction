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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# same with main

class VAE(torch.nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()


        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(2048, latent_dim)  # latent_dim *2 < inputdim x inputdim
        self.encoder_logvar = torch.nn.Linear(2048, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim),
            torch.nn.Sigmoid(),
        )


    def encode(self, x):
        v = self.encoder(x)
        mu = self.encoder_mu(v)
        logvar = self.encoder_logvar(v)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return mu, logvar, z, recon_x




def train_loss(batch_size, x, recon_x, mu_1, logvar_1, mu_2 = None, logvar_2 = None):
    if mu_2 == None and logvar_2==None:
        mu_2 = torch.zeros_like(mu_1)
        logvar_2 = torch.log(torch.ones_like(logvar_1))
    reconst_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size # nn.MSELoss()(recon_x, x) # nn.BCEWithLogitsLoss()(recon_x, x) / batch_size
    regular_loss = torch.mean(0.5 * ((1/(logvar_2.exp()))*((mu_1-mu_2).pow(2)) + (logvar_1.exp())/(logvar_2.exp())  - 1 - (logvar_1-logvar_2))) #sum 막해도되는지
    return reconst_loss, regular_loss