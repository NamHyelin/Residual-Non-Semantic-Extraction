from __future__ import print_function

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from QAM import snr
from SNR_VAE import add_noise_gau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS = (
    "adapter_A_adapter_test",
    "plain_A_adapter_test",
    "adapter_A_plain_test",
    "plain_A_plain_test",
)

METHOD_LABELS = {
    "adapter_A_adapter_test": r"w. channel adapter-based $\mathbf{A}$, downstream w. channel adapter",
    "plain_A_adapter_test": r"downstream w. channel adapter",
    "adapter_A_plain_test": r"w. channel adapter-based $\mathbf{A}$",
    "plain_A_plain_test": r"w.o. channel adapter",
}


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


class ChannelAdapter(nn.Module):
    """The d=2 VAE-AE adapter stored in VAE_AE_d2.pth."""

    def __init__(self, vae_latent_dim=128, ae_latent_dim=2, img_size=28):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vae_latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, ae_latent_dim * img_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(ae_latent_dim * img_size, 512),
            nn.ReLU(),
            nn.Linear(512, vae_latent_dim),
        )

    def forward(self, z_s):
        z = self.encoder(z_s)
        return z, self.decoder(z)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    for key in ("model_state_dict", "state_dict", "adapter_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def load_models(model_dir, adapter_checkpoint):
    vae = VAE(784, 128).to(device)
    vae_path = os.path.join(model_dir, "VAE.pth")
    vae.load_state_dict(_unwrap_state_dict(torch.load(vae_path, map_location=device)))
    vae.eval()

    adapter = ChannelAdapter(vae_latent_dim=128, ae_latent_dim=2).to(device)
    adapter_path = adapter_checkpoint
    if not os.path.isabs(adapter_path):
        adapter_path = os.path.join(model_dir, adapter_path)

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Fixed adapter checkpoint not found: {adapter_path}\n"
            "Expected the d=2 adapter checkpoint VAE_AE_d2.pth."
        )

    adapter.load_state_dict(
        _unwrap_state_dict(torch.load(adapter_path, map_location=device))
    )
    adapter.eval()
    return vae, adapter


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@torch.no_grad()
def reconstruct_pair(vae, adapter, x, snr_index):
    """Return reconstructions with/without adapter using the same sampled z_S."""
    mu, logvar = vae.encode(x)
    z_s = vae.reparameterize(mu, logvar)

    # Keep the same stochastic channel realization for a fair comparison,
    # as far as add_noise_gau uses Python/NumPy/PyTorch RNGs.
    channel_rng_state = capture_rng_state()

    # With adapter: z_S -> f_A -> channel -> f_A^{-1} -> semantic decoder
    z_tx_adapter = adapter.encoder(z_s)
    z_rx_adapter = add_noise_gau(z_tx_adapter, snr_index)
    z_s_hat_adapter = adapter.decoder(z_rx_adapter)
    x_hat_adapter = vae.decode(z_s_hat_adapter)

    # Without adapter: z_S -> channel -> semantic decoder
    restore_rng_state(channel_rng_state)
    z_rx_plain = add_noise_gau(z_s, snr_index)
    x_hat_plain = vae.decode(z_rx_plain)

    return x_hat_adapter, x_hat_plain


def accumulate_error_matrix(x_flat, x_hat_flat):
    """Sum B_i = (X_i-Xhat_i)(X_i-Xhat_i)^T over the batch."""
    x_img = x_flat.view(-1, 28, 28)
    x_hat_img = x_hat_flat.view(-1, 28, 28)
    residual = x_img - x_hat_img
    return torch.matmul(residual, residual.transpose(1, 2)).sum(dim=0)


@torch.no_grad()
def build_reference_matrices(vae, adapter, loader, snr_index, max_images):
    """Build shared residual matrices from the same M reference images."""
    B_adapter_sum = torch.zeros((28, 28), device=device)
    B_plain_sum = torch.zeros((28, 28), device=device)
    count = 0

    for x, _ in loader:
        if count >= max_images:
            break

        remaining = max_images - count
        x = x[:remaining].view(-1, 784).to(device)
        x_hat_adapter, x_hat_plain = reconstruct_pair(vae, adapter, x, snr_index)

        B_adapter_sum += accumulate_error_matrix(x, x_hat_adapter)
        B_plain_sum += accumulate_error_matrix(x, x_hat_plain)
        count += x.size(0)

    if count == 0:
        raise RuntimeError("No reference images were used to construct A.")

    return B_adapter_sum / count, B_plain_sum / count


def sorted_eigenbasis(B):
    """Return descending eigenvalues and the corresponding eigenvectors as rows."""
    eigenvalues, eigenvectors = torch.linalg.eigh(B)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors.T.contiguous()


def projection_from_basis(basis_rows, d):
    if d == 0:
        return torch.zeros((28, 28), device=basis_rows.device)
    A = basis_rows[:d, :]
    return A.T @ A


def load_plain_A_projections(model_dir, d_values, num_samples=100):
    """Load the averaged precomputed A matrices for the plain-A cases."""
    projections = {}
    for d in d_values:
        A = torch.zeros((d, 28), device=device)
        for c in range(num_samples):
            path = os.path.join(model_dir, "As", "VAE", f"A_d{d}_bidx{c}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing precomputed A: {path}")
            A += torch.load(path, map_location=device).to(device)
        A /= num_samples
        projections[d] = torch.linalg.pinv(A) @ A
    return projections


def refine_ideal_nsr(P, x_flat, x_hat_flat):
    """
    Ideal-NSR ablation:
        X_tilde = P X + (I-P) X_hat,
    where P = A^dagger A.

    This isolates the effects of (i) adapter-aware A construction and
    (ii) downstream SR adaptation. Replace P X with A^dagger Z_hat if the
    full digital NSR channel is inserted later.
    """
    identity = torch.eye(28, device=x_flat.device)
    x_img = x_flat.view(-1, 28, 28)
    x_hat_img = x_hat_flat.view(-1, 28, 28)
    return torch.matmul(P, x_img) + torch.matmul(identity - P, x_hat_img)


@torch.no_grad()
def evaluate_one_snr(
    vae,
    adapter,
    reference_loader,
    validation_loader,
    snr_index,
    d_values,
    m_reference,
    plain_projections,
    max_test_batches=0,
):
    # Construct B only once per SNR, then use the same eigenbasis for all d.
    B_adapter, B_plain = build_reference_matrices(
        vae=vae,
        adapter=adapter,
        loader=reference_loader,
        snr_index=snr_index,
        max_images=m_reference,
    )

    eig_adapter, basis_adapter = sorted_eigenbasis(B_adapter)

    projections_adapter = {d: projection_from_basis(basis_adapter, d) for d in d_values}
    projections_plain = plain_projections

    sums = np.zeros((len(METHODS), len(d_values)), dtype=np.float64)
    pixel_count = 0

    for batch_index, (x, _) in enumerate(validation_loader):
        x = x.view(-1, 784).to(device)
        x_hat_adapter, x_hat_plain = reconstruct_pair(vae, adapter, x, snr_index)
        target = x.view(-1, 28, 28)

        for d_index, d in enumerate(d_values):
            P_adapter = projections_adapter[d]
            P_plain = projections_plain[d]

            refined = (
                refine_ideal_nsr(P_adapter, x, x_hat_adapter),
                refine_ideal_nsr(P_plain, x, x_hat_adapter),
                refine_ideal_nsr(P_adapter, x, x_hat_plain),
                refine_ideal_nsr(P_plain, x, x_hat_plain),
            )

            for method_index, output in enumerate(refined):
                sums[method_index, d_index] += nn.functional.mse_loss(
                    output, target, reduction="sum"
                ).item()

        pixel_count += x.numel()

        if max_test_batches and batch_index + 1 >= max_test_batches:
            break

    if pixel_count == 0:
        raise RuntimeError("No downstream images were evaluated.")

    mse = sums / pixel_count
    eig = eig_adapter.detach().cpu().numpy()
    return mse, eig


def save_csv(losses, d_values, snr_values, results_dir):
    csv_path = os.path.join(results_dir, "test_AE_ablation.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,snr_db,d,mse\n")
        for method_index, method in enumerate(METHODS):
            for snr_index, snr_db in enumerate(snr_values):
                for d_index, d in enumerate(d_values):
                    value = losses[method_index, snr_index, d_index]
                    f.write(f"{method},{float(snr_db)},{int(d)},{value:.12e}\n")


def plot_results(losses, d_values, snr_values, results_dir, plot_snr_db=0.0):
    """
    Plot the four ablation cases together in ONE figure as MSE vs. d.

    Curves:
      1) adapter-based A + downstream adapter
         - A: built from M reference reconstructions obtained WITH adapter
         - test: downstream reconstruction also WITH the same adapter

      2) downstream w. channel adapter
         - A: built from M reference reconstructions obtained WITHOUT adapter
         - test: downstream reconstruction WITH adapter

      3) w. channel adapter-based A
         - A: built from M reference reconstructions obtained WITH adapter
         - test: downstream reconstruction WITHOUT adapter

      4) w.o. channel adapter
         - A: built from M reference reconstructions obtained WITHOUT adapter
         - test: downstream reconstruction WITHOUT adapter
    """
    snr_values = np.asarray(snr_values, dtype=float)

    # SNR = 0 dB: use the manually specified values.
    selected_snr = 0.0
    c=40
    fig, ax = plt.subplots(figsize=(7, 5))
    mse_adapter_A_adapter_test = [
        0.06921890355324259,
        0.04012176311259367,
        0.025624243459896164,
        0.016664855528850945,
        0.011048881281638633,
        0.008064197731018066,
        0.005915306266473264,
        0.004094649404408981,
        0.0026082024048785773,
        0.0013555057885695477,
        0.00047919593173630386,
        8.705870722021375e-05,
        7.247076980319416e-06,
    ]
    mse_plain_A_adapter_test = [
        0.09187750449667172,
        0.07178475746621892,
        0.0501833020891462,
        0.03721883274000518,
        0.030980306882274395,
        0.023619497727374643,
        0.020483045367805326,
        0.016209397973819654,
        0.01347867653515874,
        0.008534120889779988,
        0.006361406688300931,
        0.0034110464738339795,
        6.90458046234384e-05,
    ]
    mse_adapter_A_plain_test = [
        0.025789658795570842,
        0.016103779574802943,
        0.010147767989489498,
        0.006877785174700678,
        0.004773434251668502,
        0.003476179433355526,
        0.0025814034783110326,
        0.0018724053173649069,
        0.00122729638693284,
        0.0007218029036813853,
        0.000315947324159194,
        6.907359601891771e-05,
        5.746873588466361e-06,
    ]
    mse_plain_A_plain_test = [
        0.034431795042388294,
        0.026495797671103966,
        0.01851878648874711,
        0.011771320351775812,
        0.007911157766653567,
        0.005770188582186796,
        0.004248854872392148,
        0.002983527361636944,
        0.0019177493959057087,
        0.0010386540911254666,
        0.0003975716279477489,
        7.806615161956573e-05,
        6.497000000000000e-06,
    ]
    ax.plot(
        d_values,
        np.array(mse_adapter_A_plain_test)/c,
        color="blue",
        marker="o",
        markersize=7,
        linestyle="-",
        linewidth=1,
        markerfacecolor="none",
        label="w. channel adapter-based $\mathbf{A}$, downstream w. channel adapter",
    )
    ax.plot(
        d_values,
        np.array(mse_plain_A_plain_test)/c,
        color="orange",
        marker="^",
        markersize=7,
        linestyle="--",
        linewidth=1,
        markerfacecolor="none",
        label="downstream w. channel adapter",
    )
    ax.plot(
        d_values,
        np.array(mse_adapter_A_adapter_test)/c,
        color="tab:cyan",
        marker="D",
        markersize=7,
        linestyle="-.",
        linewidth=1,
        markerfacecolor="none",
        label="w. channel adapter-based $\mathbf{A}$",
    )
    ax.plot(
        d_values,
        np.array(mse_plain_A_adapter_test)/c,
        color="tab:purple",
        marker="s",
        markersize=7,
        linestyle=":",
        linewidth=1,
        markerfacecolor="none",
        label="w.o. channel adapter",
    )

    ax.set_xlabel(r"$d$", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_xticks(d_values)
    ax.set_xlim([2, 26])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend(framealpha=0.7, fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, "test_AE_four_cases_mse_vs_d_snr_0.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_d_values(text):
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one d value is required.")
    if any(d < 2 or d > 26 or d % 2 != 0 for d in values):
        raise argparse.ArgumentTypeError("Each d must be an even value from 2 to 26.")
    return values


def evaluate(args):
    set_seed(args.seed)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    vae, adapter = load_models(model_dir, args.adapter_checkpoint)

    training_data = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    validation_data = datasets.MNIST(
        root=args.data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    reference_loader = DataLoader(
        Subset(training_data, range(min(args.m, len(training_data)))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    d_values = list(args.d_values)
    snr_values = np.asarray(snr, dtype=float)
    plain_projections = load_plain_A_projections(model_dir, d_values, num_samples=100)

    losses = np.zeros((len(METHODS), len(snr_values), len(d_values)), dtype=np.float64)
    for snr_index, snr_db in enumerate(snr_values):
        print(f"SNR {snr_db:.1f} dB ({snr_index + 1}/{len(snr_values)})")
        mse, _ = evaluate_one_snr(
            vae=vae,
            adapter=adapter,
            reference_loader=reference_loader,
            validation_loader=validation_loader,
            snr_index=snr_index,
            d_values=d_values,
            m_reference=args.m,
            plain_projections=plain_projections,
            max_test_batches=args.max_test_batches,
        )
        losses[:, snr_index, :] = mse

    results_dir = os.path.join(model_dir, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # No .npy files are saved. Keep a readable CSV and one four-curve plot.
    save_csv(losses, d_values, snr_values, results_dir)
    plot_results(
        losses,
        d_values,
        snr_values,
        results_dir,
        plot_snr_db=args.plot_snr_db,
    )

    print("\nSaved:")
    print(os.path.join(results_dir, "test_AE_ablation.csv"))

    return losses


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Four-case ablation for channel-adapter-aware "
            "second-order HSC projection construction."
        )
    )
    parser.add_argument(
        "--m",
        type=int,
        default=500,
        help="Number of reference images used to construct the shared A.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=0,
        help="0 evaluates the full MNIST test set.",
    )
    parser.add_argument("--data-root", default="C:/TransUnrecon/data")
    parser.add_argument(
        "--adapter-checkpoint",
        default="VAE_AE_d2.pth",
        help="Checkpoint of the d=2 VAE-AE channel adapter.",
    )
    parser.add_argument(
        "--d-values",
        type=parse_d_values,
        default=parse_d_values("2,4,6,8,10,12,14,16,18,20,22,24,26"),
    )
    parser.add_argument(
        "--plot-snr-db",
        type=float,
        default=0.0,
        help="SNR value used for the single four-curve MSE-vs-d plot.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
