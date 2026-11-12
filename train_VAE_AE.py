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
from lpips_score import LPIPS_MNIST
from QAM import Ps_qam, snr
from SNR_VAE import add_noise_qam, add_noise_gau
from train_VAE import VAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(torch.nn.Module):
    def __init__(self, vae_latent_dim, ae_latent_dim, img_size=28):
        super(AutoEncoder, self).__init__()

        self.ae_latent_dim = ae_latent_dim
        self.img_size = img_size

        # Encoder: VAE latent (vae_latent_dim) -> AE latent (ae_latent_dim * img_size)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(vae_latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, ae_latent_dim * img_size),
        )

        # Decoder: AE latent (ae_latent_dim * img_size) -> VAE latent (vae_latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(ae_latent_dim * img_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, vae_latent_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon


if __name__ == "__main__":
    # Load data
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    training_data = datasets.MNIST(
        root="C:/TransUnrecon/data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    validation_data = datasets.MNIST(
        root="C:/TransUnrecon/data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(0)
    batch_size = 100
    training_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    # Parameters setting
    num_epochs = 10
    x_dim = training_data.data.shape[1] ** 2
    vae_latent_dim = 128
    vae_method = "VAE"
    learning_rate = 1e-3

    # Load pretrained VAE model
    vae_model = VAE(x_dim, vae_latent_dim).to(device)
    vae_model.load_state_dict(
        torch.load(f"C:/TransUnrecon/VAE_MNIST_2/{vae_method}.pth")
    )
    vae_model.eval()

    # Freeze VAE model
    for param in vae_model.parameters():
        param.requires_grad = False

    # Train AE for different latent dimensions
    d_values = list(range(2, 27, 2))

    for d in d_values:
        print(f"\n\n========== Training AE with d={d} ==========")

        # Create AE model
        ae_model = AutoEncoder(vae_latent_dim, d, img_size=28).to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=learning_rate)

        reconst_loss_train, reconst_loss_test = [], []

        for e in range(num_epochs):
            reconst_loss_tr = 0.0
            reconst_loss_ts = 0.0

            # Training
            for batch_idx_tr, (x, _) in enumerate(training_loader):
                ae_model.train()
                x = x.view(-1, x_dim)
                x = x.to(device)

                optimizer.zero_grad()

                # Get VAE latent representation
                with torch.no_grad():
                    mu, logvar, z, recon_x = vae_model(x)

                # Pass through AE
                ae_z, ae_recon = ae_model(mu)

                # Train one adapter to cover the complete SNR range.
                snr_index = np.random.randint(len(snr))
                noisy_ae_recon = add_noise_gau(ae_recon, snr_index)
                final_recon = vae_model.decode(noisy_ae_recon)
                loss = F.mse_loss(final_recon, x)

                reconst_loss_tr += loss.item()

                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                for batch_idx_ts, (x, _) in enumerate(validation_loader):
                    x = x.view(-1, x_dim)
                    x = x.to(device)

                    # Get VAE latent representation
                    mu, logvar, z, recon_x = vae_model(x)

                    # Pass through AE
                    ae_z, ae_recon = ae_model(mu)

                    snr_index = np.random.randint(len(snr))
                    noisy_ae_recon = add_noise_gau(ae_recon, snr_index)
                    final_recon = vae_model.decode(noisy_ae_recon)
                    loss = F.mse_loss(final_recon, x)

                    reconst_loss_ts += loss.item()

            reconst_loss_train.append(reconst_loss_tr / batch_idx_tr)
            reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)

            print(
                f"[{e} / {num_epochs}] train loss: {reconst_loss_train[-1]:.6f}, test loss: {reconst_loss_test[-1]:.6f}"
            )

        # Save AE model
        save_path = f"C:/TransUnrecon/VAE_MNIST_2/VAE_AE_d{d}.pth"
        torch.save(ae_model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

        # Save loss
        try:
            np.save(f"results/reconst_loss_train_VAE_AE_d{d}.npy", reconst_loss_train)
            np.save(f"results/reconst_loss_test_VAE_AE_d{d}.npy", reconst_loss_test)
        except:
            os.makedirs("C:/TransUnrecon/VAE_MNIST_2/results", exist_ok=True)
            np.save(
                f"C:/TransUnrecon/VAE_MNIST_2/results/reconst_loss_train_VAE_AE_d{d}.npy",
                reconst_loss_train,
            )
            np.save(
                f"C:/TransUnrecon/VAE_MNIST_2/results/reconst_loss_test_VAE_AE_d{d}.npy",
                reconst_loss_test,
            )
