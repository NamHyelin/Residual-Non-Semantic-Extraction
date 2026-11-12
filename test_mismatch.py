from __future__ import print_function

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from unexpressed import calcul_A_MNIST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_VALUES = (2, 6, 10)


class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(2048, latent_dim)
        self.encoder_logvar = nn.Linear(2048, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.encoder_mu(hidden), self.encoder_logvar(hidden)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, self.decode(z)


def unwrap_state_dict(state):
    if not isinstance(state, dict):
        return state
    for key in ("state_dict", "model_state_dict"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def load_vae(path, latent_dim=None):
    state = unwrap_state_dict(torch.load(path, map_location=device))
    if latent_dim is None:
        latent_dim = int(state["encoder_mu.weight"].shape[0])
    model = VAE(784, latent_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def save_gray(image, path):
    image = image.detach().cpu().view(28, 28).numpy()
    plt.imsave(path, np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)


def save_set(output_root, set_index, original, reconstructed, finals):
    set_dir = os.path.join(output_root, f"set_{set_index:02d}")
    os.makedirs(set_dir, exist_ok=True)
    save_gray(original, os.path.join(set_dir, "original.png"))
    save_gray(reconstructed, os.path.join(set_dir, "reconstructed.png"))
    original_image = original.view(-1, 28, 28)
    reconstructed_image = reconstructed.view(-1, 28, 28)
    with open(os.path.join(set_dir, "mse.txt"), "w", encoding="utf-8") as file:
        reconstruction_mse = nn.MSELoss()(reconstructed_image, original_image).item()
        file.write(f"reconstructed: {reconstruction_mse:.12e}\n")
    for d, image in finals.items():
        save_gray(image, os.path.join(set_dir, f"final_recon_d{d}.png"))
        final_mse = nn.MSELoss()(image, original_image).item()
        with open(os.path.join(set_dir, "mse.txt"), "a", encoding="utf-8") as file:
            file.write(f"final_recon_d{d}: {final_mse:.12e}\n")


def projection_from_A(A):
    return torch.linalg.pinv(A) @ A


def load_second_order_projections(model_dir, d_values):
    projections = {}
    for d in d_values:
        A = torch.zeros((d, 28), device=device)
        for c in range(100):
            path = os.path.join(model_dir, "As", "VAE", f"A_d{d}_bidx{c}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing SecondOrder A: {path}")
            A += torch.load(path, map_location=device).to(device)
        projections[d] = projection_from_A(A / 100.0)
    return projections


def final_reconstruction(P, original, reconstructed):
    identity = torch.eye(28, device=device)
    original = original.view(-1, 28, 28)
    reconstructed = reconstructed.view(-1, 28, 28)
    return torch.matmul(P, original) + torch.matmul(identity - P, reconstructed)


@torch.no_grad()
def run_non_second_order(model_dir, data_loader, output_root, num_sets):
    encoder_model = load_vae(os.path.join(model_dir, "VAE.pth"), latent_dim=128)
    decoder_path = os.path.join(model_dir, "VAE_FMNIST_128.pth")
    decoder_model = load_vae(decoder_path, latent_dim=128)

    for set_index, (images, _) in enumerate(data_loader):
        if set_index >= num_sets:
            break
        original = images[0:1].view(1, 784).to(device)
        mu, logvar = encoder_model.encode(original)
        z = encoder_model.reparameterize(mu, logvar)
        reconstructed = decoder_model.decode(z)
        finals = {}
        for d in D_VALUES:
            A = calcul_A_MNIST(original, reconstructed, d, True).to(device)
            finals[d] = final_reconstruction(
                projection_from_A(A), original, reconstructed
            )
        save_set(output_root, set_index + 1, original, reconstructed, finals)


def load_full_model_with_fallback(model_dir, primary_name):
    primary_path = os.path.join(model_dir, primary_name)
    try:
        return load_vae(primary_path)
    except (RuntimeError, KeyError, FileNotFoundError) as error:
        if primary_name != "VAE_FMNIST_128.pth":
            raise
        fallback_path = os.path.join(model_dir, "VAE_FMNIST_256.pth")
        print(f"Could not load {primary_name}: {error}")
        print(f"Trying fallback: {fallback_path}")
        return load_vae(fallback_path)


@torch.no_grad()
def run_second_order(model_dir, data_loader, model_name, output_root, num_sets):
    model = load_full_model_with_fallback(model_dir, model_name)
    projections = load_second_order_projections(model_dir, D_VALUES)

    for set_index, (images, _) in enumerate(data_loader):
        if set_index >= num_sets:
            break
        original = images[0:1].view(1, 784).to(device)
        _, _, _, reconstructed = model(original)
        finals = {
            d: final_reconstruction(projections[d], original, reconstructed)
            for d in D_VALUES
        }
        save_set(output_root, set_index + 1, original, reconstructed, finals)


def build_loaders(data_root, batch_size):
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    fmnist = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    return (
        DataLoader(mnist, batch_size=batch_size, shuffle=False),
        DataLoader(fmnist, batch_size=batch_size, shuffle=False),
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save VAE mismatch reconstruction sets."
    )
    parser.add_argument("--data-root", default="C:/TransUnrecon/data")
    parser.add_argument("--num-sets", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_loader, fmnist_loader = build_loaders(args.data_root, args.batch_size)
    revision_root = os.path.join(model_dir, "results", "revision")

    run_non_second_order(
        model_dir,
        mnist_loader,
        os.path.join(revision_root, "mismatch"),
        args.num_sets,
    )
    run_second_order(
        model_dir,
        fmnist_loader,
        "VAE.pth",
        os.path.join(revision_root, "diffrefer", "VAE_on_FMNIST"),
        args.num_sets,
    )
    run_second_order(
        model_dir,
        mnist_loader,
        "VAE_FMNIST_128.pth",
        os.path.join(revision_root, "diffrefer", "VAE_FMNIST_on_MNIST"),
        args.num_sets,
    )


if __name__ == "__main__":
    main(parse_args())
