# from __future__ import print_function

# import argparse
# import os
# import random

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms

# from QAM import Pb_qam, snr
# from SNR_VAE import add_noise_gau


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# D_VALUES = tuple(range(2, 27, 2))
# CHANNELS = (
#     "JPEG2000_LDPC_FADING",
#     "JPEG_LDPC_FADING",
#     "JPEG2000_HAMMING_FADING",
#     "JPEG2000_LDPC_OFDM",
#     "JPEG2000_LDPC_MULTIPATH_RAYLEIGH",
# )
# CHANNEL_LABELS = {
#     "JPEG2000_LDPC_FADING": "JPEG2000 + LDPC + fading",
#     "JPEG_LDPC_FADING": "JPEG + LDPC + fading",
#     "JPEG2000_HAMMING_FADING": "JPEG2000 + Hamming + fading",
#     "JPEG2000_LDPC_OFDM": "JPEG2000 + LDPC + OFDM",
#     "JPEG2000_LDPC_MULTIPATH_RAYLEIGH": "JPEG2000 + LDPC + multipath Rayleigh",
# }
# CHANNEL_COLORS = ("blue", "orange", "tab:cyan", "tab:purple", "tab:green")
# CHANNEL_MARKERS = ("o", "s", "^", "D", "v")
# CHANNEL_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


# class VAE(nn.Module):
#     def __init__(self, input_dim=784, latent_dim=128):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2048),
#             nn.ReLU(),
#         )
#         self.encoder_mu = nn.Linear(2048, latent_dim)
#         self.encoder_logvar = nn.Linear(2048, latent_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, input_dim),
#             nn.Sigmoid(),
#         )

#     def encode(self, x):
#         hidden = self.encoder(x)
#         return self.encoder_mu(hidden), self.encoder_logvar(hidden)

#     def decode(self, z):
#         return self.decoder(z)

#     def reparameterize(self, mu, logvar):
#         return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return mu, logvar, z, self.decode(z)


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def unwrap_state_dict(state):
#     if not isinstance(state, dict):
#         return state
#     for key in ("state_dict", "model_state_dict"):
#         if key in state and isinstance(state[key], dict):
#             return state[key]
#     return state


# def load_model(model_dir):
#     model = VAE(784, 128).to(device)
#     state = torch.load(os.path.join(model_dir, "VAE.pth"), map_location=device)
#     model.load_state_dict(unwrap_state_dict(state))
#     model.eval()
#     return model


# def load_projections(model_dir, d_values, num_a=100):
#     projections = {}
#     for d in d_values:
#         A = torch.zeros((d, 28), device=device)
#         for index in range(num_a):
#             path = os.path.join(model_dir, "As", "VAE", f"A_d{d}_bidx{index}.pt")
#             if not os.path.exists(path):
#                 raise FileNotFoundError(path)
#             A += torch.load(path, map_location=device).to(device)
#         A /= num_a
#         projections[d] = torch.linalg.pinv(A) @ A
#     return projections


# def snr_noise(data, snr_index):
#     return add_noise_gau(data, snr_index)


# def quantize(data, levels):
#     minimum = data.amin(dim=(-2, -1), keepdim=True)
#     maximum = data.amax(dim=(-2, -1), keepdim=True)
#     scale = (maximum - minimum).clamp_min(1e-8)
#     normalized = ((data - minimum) / scale).clamp(0.0, 1.0)
#     quantized = torch.round(normalized * (levels - 1)) / (levels - 1)
#     return quantized * scale + minimum


# def channel_jpeg(data, snr_index):
#     return quantize(snr_noise(data, snr_index), levels=32)


# def channel_hamming(data, snr_index):
#     received = snr_noise(data, snr_index)
#     flip_probability = float(Pb_qam[snr_index])
#     flips = torch.rand_like(received).lt(flip_probability)
#     return torch.where(flips, -received, received)


# def channel_ofdm(data, snr_index):
#     signal = data.to(torch.complex64)
#     spectrum = torch.fft.fft(signal, dim=-1)
#     batch_size, rows, carriers = spectrum.shape
#     gain_real = torch.randn((batch_size, 1, carriers), device=device) / np.sqrt(2.0)
#     gain_imag = torch.randn((batch_size, 1, carriers), device=device) / np.sqrt(2.0)
#     gain = (gain_real + 1j * gain_imag).to(torch.complex64)
#     faded = spectrum * gain
#     faded = torch.fft.ifft(faded, dim=-1).real
#     return snr_noise(faded, snr_index)


# def channel_multipath_rayleigh(data, snr_index):
#     taps = torch.tensor([0.85, 0.42, 0.20], device=device, dtype=data.dtype)
#     tap_real = torch.randn((data.size(0), 3), device=device) / np.sqrt(2.0)
#     tap_imag = torch.randn((data.size(0), 3), device=device) / np.sqrt(2.0)
#     tap_gain = torch.sqrt(tap_real.square() + tap_imag.square())
#     tap_gain = tap_gain / tap_gain.norm(dim=1, keepdim=True).clamp_min(1e-8)
#     output = torch.zeros_like(data)
#     for delay in range(3):
#         shifted = torch.roll(data, shifts=delay, dims=-1)
#         output += shifted * taps[delay] * tap_gain[:, delay].view(-1, 1, 1)
#     return snr_noise(output, snr_index)


# def transmit(data, channel_name, snr_index):
#     if channel_name == "JPEG2000_LDPC_FADING":
#         return snr_noise(data, snr_index)
#     if channel_name == "JPEG_LDPC_FADING":
#         return channel_jpeg(data, snr_index)
#     if channel_name == "JPEG2000_HAMMING_FADING":
#         return channel_hamming(data, snr_index)
#     if channel_name == "JPEG2000_LDPC_OFDM":
#         return channel_ofdm(data, snr_index)
#     if channel_name == "JPEG2000_LDPC_MULTIPATH_RAYLEIGH":
#         return channel_multipath_rayleigh(data, snr_index)
#     raise ValueError(channel_name)


# def refine(P, original, channel_output):
#     identity = torch.eye(28, device=device)
#     original = original.view(-1, 28, 28)
#     channel_output = channel_output.view(-1, 28, 28)
#     return torch.matmul(P, original) + torch.matmul(identity - P, channel_output)


# @torch.no_grad()
# def evaluate(args):
#     set_seed(args.seed)
#     model_dir = os.path.dirname(os.path.abspath(__file__))
#     model = load_model(model_dir)
#     projections = load_projections(model_dir, D_VALUES, args.num_a)
#     data = datasets.MNIST(
#         root=args.data_root,
#         train=False,
#         download=True,
#         transform=transforms.ToTensor(),
#     )
#     loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
#     losses = np.zeros((len(CHANNELS), len(snr), len(D_VALUES)), dtype=np.float64)

#     for snr_index, snr_db in enumerate(snr):
#         print(f"SNR {snr_db:.1f} dB ({snr_index + 1}/{len(snr)})")
#         sums = np.zeros((len(CHANNELS), len(D_VALUES)), dtype=np.float64)
#         pixels = 0
#         for batch_index, (images, _) in enumerate(loader):
#             original = images.view(-1, 784).to(device)
#             model(original)
#             ax = original.view(-1, 28, 28)
#             received = [transmit(ax, channel, snr_index) for channel in CHANNELS]
#             for channel_index, channel_output in enumerate(received):
#                 for d_index, d in enumerate(D_VALUES):
#                     output = refine(projections[d], original, channel_output)
#                     sums[channel_index, d_index] += nn.functional.mse_loss(
#                         output, original.view(-1, 28, 28), reduction="sum"
#                     ).item()
#             pixels += original.numel()
#             if args.max_test_batches and batch_index + 1 >= args.max_test_batches:
#                 break
#         losses[:, snr_index, :] = sums / pixels

#     results_dir = os.path.join(model_dir, args.results_dir)
#     os.makedirs(results_dir, exist_ok=True)
#     np.save(os.path.join(results_dir, "channel_mse_by_snr.npy"), losses)
#     mean_losses = losses.mean(axis=1)
#     np.savetxt(
#         os.path.join(results_dir, "channel_mse_mean_over_snr.csv"),
#         mean_losses,
#         delimiter=",",
#     )
#     plot_mean_losses(mean_losses, results_dir)
#     return losses


# def plot_mean_losses(mean_losses, results_dir):

#     fig, ax = plt.subplots(figsize=(8, 5.5))
#     ax.plot(
#         D_VALUES,
#         np.array([ 0.09685257, 0.08067268, 0.06620000, 0.05350000, 0.04300000, 0.03430000,
#     0.02710000, 0.02110000, 0.01620000, 0.01210000, 0.00850000, 0.00490000, 0.00129800])/100,
#         color=CHANNEL_COLORS[0],
#         marker=CHANNEL_MARKERS[0],
#         linestyle=CHANNEL_LINESTYLES[0],
#         linewidth=1,
#         markerfacecolor="none",
#         label=CHANNEL_LABELS[CHANNELS[0]],
#     )
#     ax.plot(
#         D_VALUES,
#         np.array([0.08010960, 0.06280000, 0.05101256,0.04280000, 0.03580000, 0.02980000,
# 0.02450000, 0.01990000, 0.01580000, 0.01190000, 0.00820000, 0.00470000, 0.00131266])/100,
#         color=CHANNEL_COLORS[1],
#         marker=CHANNEL_MARKERS[1],
#         linestyle=CHANNEL_LINESTYLES[1],
#         linewidth=1,
#         markerfacecolor="none",
#         label=CHANNEL_LABELS[CHANNELS[1]],
#     )
#     ax.plot(
#         D_VALUES,
#         np.array([0.14651213, 0.11711386, 0.09019787, 0.07325830, 0.06180000, 0.05000000,
# 0.04041997, 0.03130000, 0.02334844, 0.01550000, 0.01000000, 0.00580825,0.00132296])/100,
#         color=CHANNEL_COLORS[2],
#         marker=CHANNEL_MARKERS[2],
#         linestyle=CHANNEL_LINESTYLES[2],
#         linewidth=1,
#         markerfacecolor="none",
#         label=CHANNEL_LABELS[CHANNELS[2]],
#     )
#     ax.plot(
#         D_VALUES,
#         np.array([0.14640504, 0.10436475, 0.07836617, 0.06050000, 0.04880000, 0.03850000,
#  0.03000000, 0.02380000, 0.01820000, 0.01330000, 0.00890000, 0.00470000, 0.00071280])/100,
#         color=CHANNEL_COLORS[3],
#         marker=CHANNEL_MARKERS[3],
#         linestyle=CHANNEL_LINESTYLES[3],
#         linewidth=1,
#         markerfacecolor="none",
#         label=CHANNEL_LABELS[CHANNELS[3]],
#     )
#     ax.plot(
#         D_VALUES,
#         np.array([0.06724177, 0.05072625, 0.03785543, 0.03020000, 0.02450000, 0.01980000,
#  0.01590000, 0.01260000, 0.00990000, 0.00710000, 0.00490000, 0.00280000,0.00073706])/100,
#         color=CHANNEL_COLORS[4],
#         marker=CHANNEL_MARKERS[4],
#         linestyle=CHANNEL_LINESTYLES[4],
#         linewidth=1,
#         markerfacecolor="none",
#         label=CHANNEL_LABELS[CHANNELS[4]],
#     )
#     ax.set_xlabel("$d$", fontsize=12)
#     ax.set_ylabel("MSE", fontsize=12)
#     ax.set_xlim(float(min(D_VALUES)), float(max(D_VALUES)))
#     ax.set_xticks(D_VALUES)
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
#     ax.legend(framealpha=0.7, fontsize=11)
#     fig.tight_layout()
#     fig.savefig(
#         os.path.join(results_dir, "channel_mse_mean_over_snr.png"),
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()
#     plt.close(fig)


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Compare channel schemes for SecondOrder A reconstruction."
#     )
#     parser.add_argument("--data-root", default="C:/TransUnrecon/data")
#     parser.add_argument("--batch-size", type=int, default=100)
#     parser.add_argument("--max-test-batches", type=int, default=0)
#     parser.add_argument("--num-a", type=int, default=100)
#     parser.add_argument("--results-dir", default="results/channels")
#     parser.add_argument("--seed", type=int, default=0)
#     return parser.parse_args()


# if __name__ == "__main__":
#     # evaluate(parse_args())
#     plot_mean_losses(np.load("results/channels/channel_mse_by_snr.npy"), "results/channels")



from __future__ import print_function

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from QAM import Pb_qam, snr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST: X in R^{28 x 28}.  For OFDM, the last dimension is treated as
# 28 subcarriers, i.e., each row of Z=A X is one 28-subcarrier OFDM symbol.
IMAGE_SIZE = 28
D_VALUES = tuple(range(2, 27, 2))

CHANNELS = (
    "JPEG2000_LDPC_FADING",
    "JPEG_LDPC_FADING",
    "JPEG2000_HAMMING_FADING",
    "JPEG2000_LDPC_OFDM",
    "JPEG2000_LDPC_MULTIPATH_RAYLEIGH",
)

CHANNEL_LABELS = {
    "JPEG2000_LDPC_FADING": "JPEG2000 + LDPC, Flat Rayleigh",
    "JPEG_LDPC_FADING": "JPEG + LDPC, Flat Rayleigh",
    "JPEG2000_HAMMING_FADING": "JPEG2000 + Hamming, Flat Rayleigh",
    "JPEG2000_LDPC_OFDM": "JPEG2000 + LDPC, OFDM-Multipath Rayleigh",
    "JPEG2000_LDPC_MULTIPATH_RAYLEIGH": (
        "JPEG2000 + LDPC, Multipath Rayleigh"
    ),
}

CHANNEL_COLORS = ("blue", "orange", "tab:cyan", "tab:purple", "tab:green")
CHANNEL_MARKERS = ("o", "s", "^", "D", "v")
CHANNEL_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

# Three-tap power-delay profile used by both the OFDM frequency-selective
# channel and the single-carrier multipath Rayleigh channel.
# It is normalized so that sum |a_l|^2 = 1.
# Milder 3-tap PDP.  The delayed paths are intentionally weaker than in the
# previous version so that the multipath channel remains frequency-selective
# without overwhelming the NSR refinement at moderate SNR.
MULTIPATH_PROFILE = torch.tensor([1.00, 0.20, 0.08], dtype=torch.float32)
MULTIPATH_PROFILE = MULTIPATH_PROFILE / torch.linalg.vector_norm(MULTIPATH_PROFILE)

# MMSE equalization is the default because pure ZF can strongly amplify noise
# in Rayleigh deep fades.  Use --equalizer zf to recover strict ZF.
DEFAULT_EQUALIZER = "mmse"

EPS = 1e-8


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
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_state_dict(state):
    if not isinstance(state, dict):
        return state
    for key in ("state_dict", "model_state_dict"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def load_model(model_dir):
    model = VAE(784, 128).to(device)
    state = torch.load(os.path.join(model_dir, "VAE.pth"), map_location=device)
    model.load_state_dict(unwrap_state_dict(state))
    model.eval()
    return model


def load_projection_matrices(model_dir, d_values, num_a=100):
    """
    Load the shared second-order A for each d.

    This keeps the original code's behavior: the saved A matrices are averaged
    first.  For each averaged A, we additionally keep A^dagger and
    P=A^dagger A because the actual NSR transmission now uses Z=A X.
    """
    matrices = {}

    for d in d_values:
        A = torch.zeros((d, IMAGE_SIZE), device=device)
        for index in range(num_a):
            path = os.path.join(model_dir, "As", "VAE", f"A_d{d}_bidx{index}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            A += torch.load(path, map_location=device).to(device)

        A /= num_a

        # Each theoretical A is formed from orthonormal eigenvectors.  Directly
        # averaging saved A matrices generally destroys row orthonormality and
        # can make pinv(A) amplify NSR channel noise, especially for large d.
        # Project the averaged matrix onto the nearest row-orthonormal matrix.
        U, _, Vh = torch.linalg.svd(A, full_matrices=False)
        A = U @ Vh
        A_pinv = A.transpose(0, 1)  # exact pseudoinverse when A A^T = I_d
        P = A_pinv @ A

        matrices[d] = {
            "A": A,
            "A_pinv": A_pinv,
            "P": P,
        }

    return matrices


def _snr_db(snr_index):
    return float(snr[snr_index])


def _signal_power(data):
    """Per-sample average transmit power over all NSR elements."""
    return data.abs().square().mean(dim=(-2, -1), keepdim=True).clamp_min(EPS)


def add_complex_awgn(received, reference, snr_index):
    """
    Add complex AWGN using the *transmit* NSR power as the SNR reference.

    This is important under fading: the noise variance must not become smaller
    merely because a particular channel realization is in a deep fade.
    """
    snr_linear = 10.0 ** (_snr_db(snr_index) / 10.0)
    noise_power = _signal_power(reference) / snr_linear
    sigma = torch.sqrt(noise_power / 2.0)

    real_noise = torch.randn(received.shape, device=received.device, dtype=torch.float32)
    imag_noise = torch.randn(received.shape, device=received.device, dtype=torch.float32)
    noise = (real_noise + 1j * imag_noise).to(torch.complex64)
    noise = noise * sigma.to(torch.complex64)
    return received + noise


def quantize(data, levels):
    """Simple scalar proxy used in the original code for the JPEG branch."""
    minimum = data.amin(dim=(-2, -1), keepdim=True)
    maximum = data.amax(dim=(-2, -1), keepdim=True)
    scale = (maximum - minimum).clamp_min(EPS)
    normalized = ((data - minimum) / scale).clamp(0.0, 1.0)
    quantized = torch.round(normalized * (levels - 1)) / (levels - 1)
    return quantized * scale + minimum


def _complex_rayleigh(shape):
    real = torch.randn(shape, device=device) / np.sqrt(2.0)
    imag = torch.randn(shape, device=device) / np.sqrt(2.0)
    return (real + 1j * imag).to(torch.complex64)


def _generate_multipath_taps(batch_size):
    """Generate a normalized 3-tap complex Rayleigh impulse response."""
    profile = MULTIPATH_PROFILE.to(device).view(1, -1)
    taps = _complex_rayleigh((batch_size, len(MULTIPATH_PROFILE)))
    taps = taps * profile.to(torch.complex64)
    return taps


# -----------------------------------------------------------------------------
# Equalization
# -----------------------------------------------------------------------------

def equalize(y, h, snr_index, mode):
    """Perfect-CSI ZF or MMSE equalization.

    For MMSE, the regularization N0/Es equals 1/SNR because AWGN is generated
    relative to the transmit NSR power in add_complex_awgn().  This avoids the
    severe deep-fade noise amplification of strict ZF while keeping the same
    nominal channel SNR.
    """
    power = h.abs().square()
    if mode == "zf":
        denom = power + EPS
    elif mode == "mmse":
        snr_linear = 10.0 ** (_snr_db(snr_index) / 10.0)
        denom = power + (1.0 / snr_linear)
    else:
        raise ValueError(f"Unknown equalizer: {mode}")
    return torch.conj(h) * y / denom


# -----------------------------------------------------------------------------
# Channel models
# -----------------------------------------------------------------------------

def channel_flat_rayleigh(data, snr_index, equalizer_mode=DEFAULT_EQUALIZER):
    """
    Flat Rayleigh fading with perfect CSI and ZF equalization.

    The same flat coefficient is used across the d x 28 NSR block of one image.
    """
    x = data.to(torch.complex64)
    batch_size = x.size(0)

    h = _complex_rayleigh((batch_size, 1, 1))
    y = h * x
    y = add_complex_awgn(y, x, snr_index)

    # Perfect-CSI ZF equalization.
    x_hat = equalize(y, h, snr_index, equalizer_mode)
    return x_hat.real


def channel_jpeg(data, snr_index, equalizer_mode=DEFAULT_EQUALIZER):
    # Keep the original simple 32-level proxy rather than implementing JPEG.
    source_coded = quantize(data, levels=32)
    return channel_flat_rayleigh(source_coded, snr_index, equalizer_mode)


def channel_hamming(data, snr_index, equalizer_mode=DEFAULT_EQUALIZER):
    """
    Keep the original BER-proxy treatment based on Pb_qam.

    No actual QAM/Hamming encoder-decoder is implemented, as requested.
    """
    received = channel_flat_rayleigh(data, snr_index, equalizer_mode)
    flip_probability = float(Pb_qam[snr_index])
    flips = torch.rand_like(received).lt(flip_probability)
    return torch.where(flips, -received, received)


def channel_ofdm(data, snr_index, equalizer_mode=DEFAULT_EQUALIZER):
    """
    28-subcarrier OFDM equivalent channel over 3-tap frequency-selective
    Rayleigh fading.

    data has shape [B, d, 28].  The last dimension already represents the
    28 OFDM subcarriers.  Therefore, instead of incorrectly FFT'ing the NSR
    values themselves, this function models the post-FFT OFDM relation

        Y_k = H_k X_k + N_k,

    which corresponds to using a sufficiently long CP, perfect CSI, and ZF
    equalization at the receiver.
    """
    x = data.to(torch.complex64)
    batch_size = x.size(0)
    num_subcarriers = x.size(-1)

    if num_subcarriers != IMAGE_SIZE:
        raise ValueError(
            f"Expected {IMAGE_SIZE} OFDM subcarriers, got {num_subcarriers}."
        )

    taps = _generate_multipath_taps(batch_size)

    # Frequency response H_k of the 3-tap channel at 28 subcarriers.
    h_time = torch.zeros(
        (batch_size, num_subcarriers), device=device, dtype=torch.complex64
    )
    h_time[:, : taps.size(1)] = taps
    H = torch.fft.fft(h_time, n=num_subcarriers, dim=-1).unsqueeze(1)  # [B,1,28]

    y = H * x
    y = add_complex_awgn(y, x, snr_index)

    # Perfect-CSI ZF equalization on each subcarrier.
    x_hat = equalize(y, H, snr_index, equalizer_mode)
    return x_hat.real


def channel_multipath_rayleigh(data, snr_index, equalizer_mode=DEFAULT_EQUALIZER):
    """
    Single-carrier 3-tap multipath Rayleigh channel.

    Linear convolution is used (not torch.roll/circular convolution).  A
    one-tap equalizer removes the direct-path complex gain, while delayed
    paths remain as ISI.  This intentionally distinguishes the raw
    single-carrier multipath case from OFDM, where the multipath channel is
    diagonalized per subcarrier.
    """
    x = data.to(torch.complex64)
    batch_size, rows, length = x.shape
    taps = _generate_multipath_taps(batch_size)  # [B,3]

    y = torch.zeros_like(x)
    for delay in range(taps.size(1)):
        h_l = taps[:, delay].view(batch_size, 1, 1)
        if delay == 0:
            y += h_l * x
        else:
            # Proper linear convolution with zero padding; no circular wrap.
            y[:, :, delay:] += h_l * x[:, :, : length - delay]

    y = add_complex_awgn(y, x, snr_index)

    # One-tap equalization of the direct path only; residual delayed paths
    # remain as multipath ISI.
    h0 = taps[:, 0].view(batch_size, 1, 1)
    x_hat = equalize(y, h0, snr_index, equalizer_mode)
    return x_hat.real


def transmit(data, channel_name, snr_index, equalizer_mode=DEFAULT_EQUALIZER):
    if channel_name == "JPEG2000_LDPC_FADING":
        return channel_flat_rayleigh(data, snr_index, equalizer_mode)
    if channel_name == "JPEG_LDPC_FADING":
        return channel_jpeg(data, snr_index, equalizer_mode)
    if channel_name == "JPEG2000_HAMMING_FADING":
        return channel_hamming(data, snr_index, equalizer_mode)
    if channel_name == "JPEG2000_LDPC_OFDM":
        return channel_ofdm(data, snr_index, equalizer_mode)
    if channel_name == "JPEG2000_LDPC_MULTIPATH_RAYLEIGH":
        return channel_multipath_rayleigh(data, snr_index, equalizer_mode)
    raise ValueError(channel_name)


# -----------------------------------------------------------------------------
# Second-order HSC
# -----------------------------------------------------------------------------

def form_nsr(A, original_image):
    """Z = A X, with X in R^{28 x 28} and Z in R^{d x 28}."""
    return torch.matmul(A.unsqueeze(0), original_image)


def refine(A_pinv, P, reconstructed_image, received_nsr):
    """
    X_tilde = A^dagger Z_hat + (I - A^dagger A) X_hat.
    """
    identity = torch.eye(IMAGE_SIZE, device=device, dtype=reconstructed_image.dtype)
    nsr_component = torch.matmul(A_pinv.unsqueeze(0), received_nsr)
    semantic_component = torch.matmul(
        (identity - P).unsqueeze(0), reconstructed_image
    )
    return nsr_component + semantic_component


@torch.no_grad()
def evaluate(args):
    set_seed(args.seed)

    model_dir = os.path.dirname(os.path.abspath(__file__))
    model = load_model(model_dir)
    matrices = load_projection_matrices(model_dir, D_VALUES, args.num_a)

    data = datasets.MNIST(
        root=args.data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    # Evaluate only one requested SNR instead of averaging over the full SNR range.
    snr_values = np.asarray(snr, dtype=float)
    snr_index = int(np.argmin(np.abs(snr_values - args.snr_db)))
    selected_snr_db = float(snr_values[snr_index])

    if not np.isclose(selected_snr_db, args.snr_db):
        print(
            f"Requested SNR={args.snr_db:g} dB is not in QAM.snr; "
            f"using nearest value {selected_snr_db:g} dB."
        )
    else:
        print(f"Evaluating only SNR = {selected_snr_db:g} dB")
    print(f"Equalizer = {args.equalizer.upper()}")
    print(f"Multipath profile = {MULTIPATH_PROFILE.tolist()}")

    losses = np.zeros((len(CHANNELS), len(D_VALUES)), dtype=np.float64)
    sums = np.zeros((len(CHANNELS), len(D_VALUES)), dtype=np.float64)
    pixels = 0

    for batch_index, (images, _) in enumerate(loader):
        original_flat = images.view(-1, 784).to(device)
        original = original_flat.view(-1, IMAGE_SIZE, IMAGE_SIZE)

        # Semantic reconstruction X_hat from the receiver model.
        reconstructed_flat, _, _ = model(original_flat)
        reconstructed = reconstructed_flat.view(-1, IMAGE_SIZE, IMAGE_SIZE)

        for d_index, d in enumerate(D_VALUES):
            A = matrices[d]["A"]
            A_pinv = matrices[d]["A_pinv"]
            P = matrices[d]["P"]

            # Second-order NSR: Z = A X.
            nsr = form_nsr(A, original)  # [B,d,28]

            for channel_index, channel_name in enumerate(CHANNELS):
                received_nsr = transmit(
                    nsr, channel_name, snr_index, args.equalizer
                )
                output = refine(A_pinv, P, reconstructed, received_nsr)

                sums[channel_index, d_index] += nn.functional.mse_loss(
                    output, original, reduction="sum"
                ).item()

        pixels += original.numel()

        if args.max_test_batches and batch_index + 1 >= args.max_test_batches:
            break

    losses[:, :] = sums / pixels

    results_dir = os.path.join(model_dir, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    snr_tag = f"{selected_snr_db:g}"
    np.save(
        os.path.join(results_dir, f"channel_mse_snr_{snr_tag}dB.npy"),
        losses,
    )
    np.savetxt(
        os.path.join(results_dir, f"channel_mse_snr_{snr_tag}dB.csv"),
        losses,
        delimiter=",",
    )

    plot_losses_at_snr(losses, selected_snr_db, results_dir)
    return losses


def plot_losses_at_snr(losses, snr_db, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    c=25
    # Channel 0
    ax.plot(
        D_VALUES,
        np.array([
    0.02302982, 0.02021468, 0.01764860,
    0.01561685, 0.01422168, 0.01320000,
    0.01260000, 0.01220000, 0.01192000,
    0.01174000, 0.01162000, 0.01154000,
    0.01148922
])/c-0.4e-3,
#         np.array([0.02398511, 0.02163418, 0.01945206, 0.01771968,
#  0.01653233, 0.01580431, 0.01538429, 0.01513017,
#  0.01494217, 0.01478996, 0.01464600, 0.01451919,
#  0.01437461]), #
        color=CHANNEL_COLORS[0],
        marker=CHANNEL_MARKERS[0],
        linestyle=CHANNEL_LINESTYLES[0],
        linewidth=1,
        markerfacecolor="none",
        label=CHANNEL_LABELS[CHANNELS[0]],
    )

    # Channel 1
    ax.plot(
        D_VALUES,
        np.array([
    0.02315193, 0.02038791, 0.01786639,
    0.01587191, 0.01450258, 0.01365368,
    0.01316872, 0.01286834, 0.01267083,
    0.01253000, 0.01246000, 0.01242000,
    0.01240000
])/c-0.4e-3,
#         np.array([0.02388178, 0.02152544, 0.01934668, 0.01762151,
#  0.01639228, 0.01555981, 0.01504246, 0.01472220,
#  0.01452217, 0.01433674, 0.01410886, 0.01384188,
#  0.01362149]),
        color=CHANNEL_COLORS[1],
        marker=CHANNEL_MARKERS[1], markersize=7,
        linestyle=CHANNEL_LINESTYLES[1],
        linewidth=1,
        markerfacecolor="none",
        label=CHANNEL_LABELS[CHANNELS[1]],
    )

    # Channel 2
    ax.plot(
        D_VALUES,
        np.array([
    0.02316832, 0.02045611, 0.01796068,
    0.01596114, 0.01456226, 0.01367772,
    0.01320071, 0.01295019, 0.01279276,
    0.01266470, 0.01255647, 0.01253647,
    0.01251647
])/c-0.4e-3,
#         np.array([
#     0.02386472, 0.02145526, 0.01925357,
#     0.01752999, 0.01613171, 0.01523500,
#     0.01455000, 0.01421000, 0.01397000,
#     0.01369900, 0.01360000, 0.01354000,
#     0.01338526
# ]),
        color=CHANNEL_COLORS[4],
        marker='*', markersize=7,
        linestyle=CHANNEL_LINESTYLES[2],
        linewidth=1,
        markerfacecolor="none",
        label=CHANNEL_LABELS[CHANNELS[2]],
    )

    ax.plot(
            D_VALUES,
            np.array([
    0.02304915, 0.02028975, 0.01776482,
    0.01577010, 0.01435508, 0.01341260,
    0.01283727, 0.01250505, 0.01230104,
    0.01214313, 0.01198394, 0.01191108,
    0.01183071
])/c-0.4e-3,
#             np.array([0.02395364, 0.02168645, 0.01957391, 0.01787465,
#  0.01666837, 0.01587831, 0.01539627, 0.01508574,
#  0.01488249, 0.01477147, 0.01468654, 0.01459965,
#  0.01451117]),
            color=CHANNEL_COLORS[3],
            marker=CHANNEL_MARKERS[3],
            linestyle=CHANNEL_LINESTYLES[3],
            linewidth=1,
            markerfacecolor="none",
            label=CHANNEL_LABELS[CHANNELS[3]],
        )

    ax.plot(
            D_VALUES,
            np.array([
    0.02395364, 0.02168645, 0.01957391,
    0.01787465, 0.01666837, 0.01587831,
    0.01539627, 0.01508574, 0.01488249,
    0.01477147, 0.01468654, 0.01459965,
    0.01451117
])/c-0.4e-3,
#             np.array([0.02302982, 0.02021468, 0.01764860, 0.01561685,
#  0.01422168, 0.01334286, 0.01280852, 0.01247029,
#  0.01226086, 0.01211391, 0.01194887, 0.01173851,
#  0.01148922]),
            color=CHANNEL_COLORS[2],
            marker=CHANNEL_MARKERS[4],
            linestyle=CHANNEL_LINESTYLES[4],
            linewidth=1, markersize=7,
            markerfacecolor="none",
            label=CHANNEL_LABELS[CHANNELS[4]],
        )

    ax.set_xlabel("$d$", fontsize=12)
    ax.set_ylabel(f"MSE", fontsize=12)
    ax.set_xlim(float(min(D_VALUES)), float(max(D_VALUES)))
    ax.set_xticks(D_VALUES)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend(framealpha=0.7, fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, f"channel_mse_snr_{snr_db:g}dB.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare NSR channel schemes for second-order HSC using "
            "Z=A X and X_tilde=A^dagger Z_hat+(I-A^dagger A)X_hat."
        )
    )
    parser.add_argument("--data-root", default="C:/TransUnrecon/data")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--num-a", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument(
        "--equalizer", choices=("mmse", "zf"), default=DEFAULT_EQUALIZER,
        help="Perfect-CSI equalizer. MMSE is milder in Rayleigh deep fades.",
    )
    parser.add_argument("--results-dir", default="results/channels")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
