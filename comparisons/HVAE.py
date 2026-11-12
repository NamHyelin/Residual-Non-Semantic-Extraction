import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2048)
        self.fc4_mu = nn.Linear(2048, latent_dim)
        self.fc4_logvar = nn.Linear(2048, latent_dim)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        mu = self.fc4_mu(h3)
        logvar = self.fc4_logvar(h3)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        x_recon = torch.sigmoid(self.fc4(h3))
        x_recon = x_recon.view(-1, 1, 28, 28)
        return x_recon


class HierarchicalVAE(nn.Module):
    def __init__(self, latent_dim1=128, latent_dim2=64):
        super(HierarchicalVAE, self).__init__()
        self.encoder1 = Encoder(latent_dim=latent_dim1)
        self.encoder2 = Encoder(latent_dim=latent_dim2)
        self.decoder = Decoder(latent_dim=latent_dim1 + latent_dim2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode to get two levels of latent variables
        mu1, logvar1 = self.encoder1(x)
        mu2, logvar2 = self.encoder2(x)

        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        # Skip connection: concatenate latent levels
        z_combined = torch.cat([z1, z2], dim=1)

        x_recon = self.decoder(z_combined)
        return x_recon, mu1, logvar1, mu2, logvar2





class PriorZ1GivenZ2(nn.Module):
    def __init__(self, z2_dim, z1_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z2_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z1_dim * 2)  # outputs both μ and logσ²
        )

    def forward(self, z2):
        stats = self.fc(z2)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar




class HierarchicalVAE(nn.Module):
    def __init__(self, z1_dim, z2_dim):
        super().__init__()
        self.encoder_z2 = Encoder(latent_dim=z2_dim)              # q(z2 | x)
        self.encoder_z1 = nn.Sequential(                          # q(z1 | z2, x)
            nn.Linear(2048 + z2_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z1_dim * 2)                            # μ, logσ²
        )
        self.decoder = Decoder(latent_dim=z1_dim + z2_dim)       # p(x | z1, z2)
        self.prior_z1 = PriorZ1GivenZ2(z2_dim, z1_dim)           # p(z1 | z2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        x_flat = x.view(-1, 784)

        # q(z2 | x)
        mu2, logvar2 = self.encoder_z2(x)
        z2 = self.reparameterize(mu2, logvar2)

        # q(z1 | z2, x)
        h_x = F.relu(self.encoder_z2.fc3(F.relu(self.encoder_z2.fc2(F.relu(self.encoder_z2.fc1(x_flat))))))
        h_combined = torch.cat([h_x, z2], dim=1)
        stats1 = self.encoder_z1(h_combined)
        mu1, logvar1 = stats1.chunk(2, dim=-1)
        z1 = self.reparameterize(mu1, logvar1)

        # p(z1 | z2)
        prior_mu1, prior_logvar1 = self.prior_z1(z2)

        # Reconstruction
        x_recon = self.decoder(torch.cat([z1, z2], dim=1))

        return x_recon, mu1, logvar1, mu2, logvar2, prior_mu1, prior_logvar1




def hierarchical_vae_loss(x_recon, x, mu1, logvar1, mu2, logvar2, prior_mu1, prior_logvar1):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL[q(z2|x) || p(z2)] — standard Normal
    kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())

    # KL[q(z1|z2,x) || p(z1|z2)] — both learned Gaussians
    log_std_q = 0.5 * logvar1
    log_std_p = 0.5 * prior_logvar1
    kl1 = torch.sum(
        log_std_p - log_std_q +
        (torch.exp(logvar1) + (mu1 - prior_mu1).pow(2)) / torch.exp(prior_logvar1) - 1,
        dim=1
    )
    kl1 = 0.5 * kl1.sum()

    return recon_loss + kl1 + kl2


def train(num_epochs, model, training_loader, optimizer, method):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx_tr, (x, _) in enumerate(training_loader):
            x = x.to(device)
            x_recon, mu1, logvar1, mu2, logvar2, prior_mu1, prior_logvar1 = model(x)
            loss = hierarchical_vae_loss(x_recon, x, mu1, logvar1, mu2, logvar2, prior_mu1, prior_logvar1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_loader.dataset):.4f}")
        torch.save(model.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/comparisons/' + method + '.pth')


def test(model, validation_loader, method):
    model.eval()

    from SNR_VAE import add_noise_gau
    from QAM import snr
    losses = np.zeros((len(snr),3))
    with torch.no_grad():
        for batch_idx_ts, (x, _) in enumerate(validation_loader):
            # Encode to z2
            x_test= x.to(device)
            mu2, logvar2 = model.encoder_z2(x_test)
            z2 = model.reparameterize(mu2, logvar2)

            # Option 1: output with only z1, set z2 = 0
            z2_zero = torch.zeros_like(z2)
            h_x = F.relu(model.encoder_z2.fc3(F.relu(model.encoder_z2.fc2(F.relu(model.encoder_z2.fc1(x_test.view(-1, 784)))))))
            h_combined = torch.cat([h_x, z2_zero], dim=1)
            mu1_only, logvar1_only = model.encoder_z1(h_combined).chunk(2, dim=-1)
            z1_only = model.reparameterize(mu1_only, logvar1_only)
            out1 = model.decoder(torch.cat([z1_only, z2_zero], dim=1))

            # Option 2: output with z1 and z2 (standard)
            h_combined = torch.cat([h_x, z2], dim=1)
            mu1, logvar1 = model.encoder_z1(h_combined).chunk(2, dim=-1)
            z1 = model.reparameterize(mu1, logvar1)
            out2 = model.decoder(torch.cat([z1, z2], dim=1))

            # Option 3: full generation from prior
            z2_sampled = torch.randn_like(z2)
            prior_mu1, prior_logvar1 = model.prior_z1(z2_sampled)
            z1_sampled = model.reparameterize(prior_mu1, prior_logvar1)
            out3 = model.decoder(torch.cat([z1_sampled, z2_sampled], dim=1))

    return out1, out2, out3




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
    num_epochs = 60  # 0
    num_warmup_epochs = 0  # 50
    x_dim = training_data.data.shape[1] ** 2
    hidden_dim = 128  # 하고싶은거의 반값을 넣으면됨
    learning_rate = 1e-3
    method = f'HJSCC'


    model = HierarchicalVAE(z1_dim=hidden_dim, z2_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    train(num_epochs, model, training_loader, optimizer, method)
    test(model, validation_loader, method)




