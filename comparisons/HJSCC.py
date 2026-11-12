import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F


class JSCCEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, z):
        return self.fc(z)  # output: s_l (symbols)


class JSCCDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # outputs mu, logvar
        )

    def forward(self, s_tilde):
        out = self.fc(s_tilde)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar





class HierarchicalEncoder3(nn.Module):
    def __init__(self, z_dims=[128, 128, 128]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU()
        )
        self.fc_mu3 = nn.Linear(2048, z_dims[2])
        self.fc_logvar3 = nn.Linear(2048, z_dims[2])

        self.fc_mu2 = nn.Linear(2048 + z_dims[2], z_dims[1])
        self.fc_logvar2 = nn.Linear(2048 + z_dims[2], z_dims[1])

        self.fc_mu1 = nn.Linear(2048 + z_dims[2] + z_dims[1], z_dims[0])
        self.fc_logvar1 = nn.Linear(2048 + z_dims[2] + z_dims[1], z_dims[0])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        x_flat = x.view(-1, 784)
        h = self.shared(x_flat)

        mu3 = self.fc_mu3(h)
        logvar3 = self.fc_logvar3(h)
        z3 = self.reparameterize(mu3, logvar3)

        h2 = torch.cat([h, z3], dim=1)
        mu2 = self.fc_mu2(h2)
        logvar2 = self.fc_logvar2(h2)
        z2 = self.reparameterize(mu2, logvar2)

        h1 = torch.cat([h, z3, z2], dim=1)
        mu1 = self.fc_mu1(h1)
        logvar1 = self.fc_logvar1(h1)
        z1 = self.reparameterize(mu1, logvar1)

        return z1, mu1, logvar1, z2, mu2, logvar2, z3, mu3, logvar3



class HierarchicalDecoder3(nn.Module):
    def __init__(self, z_dims=[128, 128, 128]):
        super().__init__()
        self.decode1 = nn.Sequential(  # z1 only
            nn.Linear(z_dims[0], 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
        self.decode2 = nn.Sequential(  # z1 + z2
            nn.Linear(z_dims[0] + z_dims[1], 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
        self.decode3 = nn.Sequential(  # z1 + z2 + z3
            nn.Linear(sum(z_dims), 2048),
            nn.ReLU(),
            nn.Linear(2048, 784),
            nn.Sigmoid()
        )

    def forward(self, z1, z2, z3):
        x1 = self.decode1(z1)
        x2 = self.decode2(torch.cat([z1, z2], dim=1))
        x3 = self.decode3(torch.cat([z1, z2, z3], dim=1))
        return x1.view(-1, 1, 28, 28), x2.view(-1, 1, 28, 28), x3.view(-1, 1, 28, 28)



def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1,
        dim=1
    ).mean()




def train(num_epochs, jscc_encoders, jscc_decoders, encoder, decoder, training_loader, optimizer, method):
    for epoch in range(num_epochs):
        model_loss = 0
        kl_weight = min(1.0, epoch / 10.0)  # KL annealing over first 10 epochs

        for batch_idx, (x, _) in enumerate(training_loader):
            x = x.to(device)

            # === Encode
            z1, mu1, logvar1, z2, mu2, logvar2, z3, mu3, logvar3 = encoder(x)
            zs = [z1, z2, z3]
            mus_q = [mu1, mu2, mu3]
            logvars_q = [logvar1, logvar2, logvar3]

            z_hats = []
            mus_p = []
            logvars_p = []

            # === JSCC transmission for each latent
            for i in range(3):
                s = jscc_encoders[i](zs[i])
                if 'noise' in method:
                    from QAM import snr
                    from SNR_VAE import add_noise_gau
                    ber_index = np.random.randint(1, len(snr) + 1)
                    s_tilde = add_noise_gau(s, ber_index - 1)
                else:
                    s_tilde = s

                mu_hat, logvar_hat = jscc_decoders[i](s_tilde)
                logvar_hat = torch.clamp(logvar_hat, min=-10, max=10)  # stability
                z_hat = encoder.reparameterize(mu_hat, logvar_hat)

                z_hats.append(z_hat)
                mus_p.append(mu_hat)
                logvars_p.append(logvar_hat)

            # === Decode using original z's and JSCC-decoded ẑ's
            x_hat1, x_hat2, x_hat3 = decoder(zs[0], zs[1], zs[2])
            x_hatH1, x_hatH2, x_hatH3 = decoder(z_hats[0], z_hats[1], z_hats[2])

            # === Compute Loss
            alpha = 1.0
            lambda_ = 10.0
            loss = 0.0

            # KL divergence (rate)
            for i in range(3):
                rate_i = kl_divergence(mus_q[i], logvars_q[i], mus_p[i], logvars_p[i])
                loss += kl_weight * alpha * rate_i

            # Distortion from all levels
            loss += lambda_ * (
                F.mse_loss(x, x_hat1) + F.mse_loss(x, x_hatH1) +
                F.mse_loss(x, x_hat2) + F.mse_loss(x, x_hatH2) +
                F.mse_loss(x, x_hat3) + F.mse_loss(x, x_hatH3)
            )

            # === Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_loss += loss.item()

        # === Logging
        print(f"Epoch {epoch + 1:03d} | Loss: {model_loss / len(training_loader):.4f} | KL Weight: {kl_weight:.2f}")

        # === Save model
        save_dir = 'C:/TransUnrecon/VAE_MNIST_2/comparisons/'
        torch.save(encoder.state_dict(), f'{save_dir}/{method}_encoder.pth')
        torch.save(decoder.state_dict(), f'{save_dir}/{method}_decoder.pth')
        for i, (enc, dec) in enumerate(zip(jscc_encoders, jscc_decoders), start=1):
            torch.save(enc.state_dict(), f'{save_dir}/{method}_jsccencoder_{i}.pth')
            torch.save(dec.state_dict(), f'{save_dir}/{method}_jsccdecoder_{i}.pth')

        if epoch==num_epochs-1:
            print('hi')


# def train(num_epochs, jscc_encoders, jscc_decoders, encoder, decoder, training_loader, optimizer, method):
#     for epoch in range(num_epochs):
#         model_loss = 0
#         for batch_idx_tr, (x, _) in enumerate(training_loader):
#             x = x.to(device)
#
#             # Encoder: get latents
#             z1, mu1, logvar1, z2, mu2, logvar2, z3, mu3, logvar3 = encoder(x)
#
#             zs = [z1, z2, z3]
#             mus = [mu1, mu2, mu3]
#             logvars = [logvar1, logvar2, logvar3]
#             z_hats = []
#
#             # Encode → Channel → Decode → Resample for all levels
#             for i in range(3):
#                 s = jscc_encoders[i](zs[i])
#                 if 'noise' in method:
#                     from QAM import snr
#                     from SNR_VAE import add_noise_gau
#                     ber_index = np.random.randint(1, len(snr) + 1)
#                     s_tilde = add_noise_gau(s, ber_index-1)
#                 else:
#                     s_tilde = s
#                 mu_hat, logvar_hat = jscc_decoders[i](s_tilde)
#                 z_hat = encoder.reparameterize(mu_hat, logvar_hat)
#                 z_hats.append((z_hat, mu_hat, logvar_hat))
#
#             # Decode 3 outputs
#             x_hat1, x_hat2, x_hat3 = decoder(zs[0], zs[1], zs[2])
#             x_hatH1, x_hatH2, x_hatH3 = decoder(*[z[0] for z in z_hats])
#
#
#             # Loss: weighted sum of distortion + rate for 3 levels
#             def log_gaussian(z, mu, logvar):
#                 logvar = torch.clamp(logvar, min=-10, max=10)
#                 return -0.5 * ((z - mu) ** 2 / logvar.exp() + logvar + torch.log(torch.tensor(2 * torch.pi)).to(device))
#
#
#             loss = 0
#             alpha = 1
#             lambda_ = 1
#             for i in range(3):
#                 z_i, mu_hat, logvar_hat = zs[i], z_hats[i][1], z_hats[i][2]
#                 rate_i = -alpha * log_gaussian(z_i, mu_hat, logvar_hat).sum(dim=1).mean()
#                 loss += rate_i
#
#             # Distortion: each level
#             loss += lambda_ * (
#                     F.mse_loss(x, x_hat1) + F.mse_loss(x, x_hatH1) +
#                     F.mse_loss(x, x_hat2) + F.mse_loss(x, x_hatH2) +
#                     F.mse_loss(x, x_hat3) + F.mse_loss(x, x_hatH3)
#             )
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             model_loss += loss.item()
#         print(f"Epoch {epoch + 1}, Loss: {model_loss / len(training_loader):.4f}")
#
#         # Save encoder and decoder
#         torch.save(encoder.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/comparisons/{method}_encoder.pth')
#         torch.save(decoder.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/comparisons/{method}_decoder.pth')
#
#         # Save each JSCC encoder and decoder (z1, z2, z3)
#         for i, (enc, dec) in enumerate(zip(jscc_encoders, jscc_decoders), start=1):
#             torch.save(enc.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/comparisons/{method}_jsccencoder_{i}.pth')
#             torch.save(dec.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/comparisons/{method}_jsccdecoder_{i}.pth')


import matplotlib.pyplot as plt

def test(jscc_encoders, jscc_decoders, encoder, decoder, validation_loader, method):
    encoder.eval()
    decoder.eval()
    for e, d in zip(jscc_encoders, jscc_decoders):
        e.eval()
        d.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(validation_loader):
            x = x.to(device)

            # Get latent codes
            z1, _, _, z2, _, _, z3, _, _ = encoder(x)

            zs = [z1, z2, z3]
            z_hats = []

            for i in range(3):
                s = jscc_encoders[i](zs[i])
                if 'noise' in method:
                    from QAM import snr
                    from SNR_VAE import add_noise_gau
                    ber_index = np.random.randint(1, len(snr) + 1)
                    s_tilde = add_noise_gau(s, ber_index - 1)
                else:
                    s_tilde = s
                mu_hat, logvar_hat = jscc_decoders[i](s_tilde)
                z_hat = encoder.reparameterize(mu_hat, logvar_hat)
                z_hats.append(z_hat)

            # Decode using reconstructed latent vectors
            x_hat1, x_hat2, x_hat3 = decoder(z_hats[0], z_hats[1], z_hats[2])

            # Plot results
            def show_images(images, title):
                images = images[:8].cpu().numpy()
                fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
                for i, img in enumerate(images):
                    axes[i].imshow(img.squeeze(), cmap='gray')
                    axes[i].axis('off')
                plt.suptitle(title)
                plt.show()

            show_images(x, "Original")
            show_images(x_hat1, "Reconstruction from z1")
            show_images(x_hat2, "Reconstruction from z1 + z2")
            show_images(x_hat3, "Reconstruction from z1 + z2 + z3")


if __name__ == "__main__":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))  # , transforms.Normalize((0.1307), (0.3081))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))  # , transforms.Normalize((0.1307), (0.3081))


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
    num_epochs = 100  # 0
    x_dim = training_data.data.shape[1] ** 2
    hidden_dim = 128  # 하고싶은거의 반값을 넣으면됨
    learning_rate = 1e-3
    method = f'HJSCC'

    encoder = HierarchicalEncoder3([hidden_dim,hidden_dim,hidden_dim]).to(device)
    decoder = HierarchicalDecoder3([hidden_dim,hidden_dim,hidden_dim]).to(device)

    # JSCC modules for each latent level
    jscc_encoders = [JSCCEncoder(hidden_dim).to(device) for _ in range(3)]
    jscc_decoders = [JSCCDecoder(hidden_dim).to(device) for _ in range(3)]

    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    for enc, dec in zip(jscc_encoders, jscc_decoders):
        params += list(enc.parameters()) + list(dec.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    train(num_epochs, jscc_encoders, jscc_decoders, encoder, decoder, training_loader, optimizer, method)
    test(jscc_encoders, jscc_decoders, encoder, decoder, validation_loader, method)