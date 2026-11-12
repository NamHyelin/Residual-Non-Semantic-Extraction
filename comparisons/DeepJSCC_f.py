import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from datetime import datetime
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from datetime import datetime


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ConvGDNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, inverse=False):
        super(ConvGDNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.act = nn.Sigmoid() if inverse else nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class Encoder(nn.Module):
    def __init__(self, conv_depth):
        super(Encoder, self).__init__()
        num_filters = 256
        self.layers = nn.Sequential(
            ConvGDNLayer(1, num_filters, 9, 2),
            nn.PReLU(),
            ConvGDNLayer(num_filters, num_filters, 5, 2),
            nn.PReLU(),
            ConvGDNLayer(num_filters, num_filters, 5, 1),
            nn.PReLU(),
            ConvGDNLayer(num_filters, num_filters, 5, 1),
            nn.PReLU(),
            nn.Conv2d(num_filters, conv_depth, 5, stride=1, padding=2)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, n_channels):
        super(Decoder, self).__init__()
        num_filters = 256
        self.layers = nn.Sequential(
            ConvGDNLayer(n_channels, num_filters, 5, 1, inverse=True),
            nn.PReLU(),
            ConvGDNLayer(num_filters, num_filters, 5, 1, inverse=True),
            nn.PReLU(),
            ConvGDNLayer(num_filters, num_filters, 5, 1, inverse=True),
            nn.PReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(num_filters, 1, 9, stride=2, padding=4, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)





class NBatchLogger:
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self._start_time = time.time()

    def log(self, logs):
        self.step += 1
        for k, v in logs.items():
            self.metric_cache[k] = self.metric_cache.get(k, 0) + v

        if self.step % self.display == 0:
            cur_time = time.time()
            duration = cur_time - self._start_time
            self._start_time = cur_time
            sec_per_step = duration / self.display

            metrics_log = ""
            for k, v in self.metric_cache.items():
                val = v / self.display
                metrics_log += f" - {k}: {val:.4f}" if abs(val) > 1e-3 else f" - {k}: {val:.4e}"

            print(f"{datetime.now()} step: {self.step} {metrics_log} - {sec_per_step:.3f} sec/step")
            self.metric_cache.clear()


class PSNRsVar:
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        psnr_vals = self._psnr(y_true, y_pred)
        samples = psnr_vals.detach().cpu().numpy()
        batch_count = samples.size
        batch_mean = np.mean(samples)
        batch_var = np.var(samples)

        new_count = self.count + batch_count
        new_mean = (self.count * self.mean + batch_count * batch_mean) / new_count
        new_var = (
            (self.count * (self.var + (self.mean - new_mean)**2)) +
            (batch_count * (batch_var + (batch_mean - new_mean)**2))
        ) / new_count

        self.count = new_count
        self.mean = new_mean
        self.var = new_var

    def result(self):
        return self.var

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.var = 0.0

    def _psnr(self, x, y):
        mse = F.mse_loss(x, y, reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        return 10 * torch.log10(1.0 / mse)


class TargetPSNRsHistogram:
    def __init__(self, min_psnr=20, max_psnr=45, step=1):
        self.bins_labels = np.arange(min_psnr, max_psnr + 1, step)
        self.bins = np.zeros_like(self.bins_labels, dtype=np.int32)

    def update(self, y_true, y_pred):
        psnrs = self._psnr(y_true, y_pred).detach().cpu().numpy()
        for i, bin_label in enumerate(self.bins_labels):
            self.bins[i] += np.sum(psnrs >= bin_label)

    def result(self):
        return self.bins

    def reset(self):
        self.bins = np.zeros_like(self.bins_labels, dtype=np.int32)

    def _psnr(self, x, y):
        mse = F.mse_loss(x, y, reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        return 10 * torch.log10(1.0 / mse)


def real_awgn(x, stddev):
    noise = torch.randn_like(x) * stddev
    return x + noise


def fading(x, stddev, h=None):
    batch_size = x.size(0)
    if h is None:
        real = torch.randn(batch_size, 1, device=x.device) / np.sqrt(2)
        imag = torch.randn(batch_size, 1, device=x.device) / np.sqrt(2)
        h = torch.complex(real, imag)

    awgn_real = torch.randn_like(x) / np.sqrt(2)
    awgn_imag = torch.randn_like(x) / np.sqrt(2)
    awgn = torch.complex(awgn_real, awgn_imag)

    x_complex = torch.view_as_complex(x.view(x.size(0), -1, 2))
    y = h.unsqueeze(-1) * x_complex + stddev * awgn
    y = torch.view_as_real(y).view_as(x)
    return y, h


def phase_invariant_fading(x, stddev, h=None):
    batch_size = x.size(0)
    if h is None:
        n1 = torch.randn(batch_size, 1, device=x.device) / np.sqrt(2)
        n2 = torch.randn(batch_size, 1, device=x.device) / np.sqrt(2)
        h = torch.sqrt(n1 ** 2 + n2 ** 2)

    awgn = torch.randn_like(x) * (stddev / np.sqrt(2))
    y = h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + awgn
    return y, h


class OutputsCombiner(nn.Module):
    def __init__(self):
        super(OutputsCombiner, self).__init__()
        self.conv1 = nn.Conv2d(2, 48, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(48, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_prev, residual):
        try:
            residual = torch.mean(residual.view(3,100,1,28,28), 0)
        except:
            residual = torch.mean(residual.view(4, 100, 1, 28, 28), 0)
        x = torch.cat((img_prev, residual), dim=1)
        x = self.prelu1(self.conv1(x))
        return self.sigmoid(self.conv2(x))


class Channel(nn.Module):
    def __init__(self, channel_type, channel_snr):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def forward(self, encoded_img, prev_h=None):
        batch_size = encoded_img.size(0)
        z = encoded_img.view(batch_size, -1)
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        if self.channel_type == "awgn":
            dim_z = z.size(1)
            z_in = torch.sqrt(torch.tensor(dim_z, dtype=torch.float32, device=z.device)) * F.normalize(z, p=2, dim=1)
            z_out = real_awgn(z_in, noise_stddev)
            h = torch.ones_like(z_in)

        elif self.channel_type == "fading":
            dim_z = z.size(1) // 2
            z_real = z[:, :dim_z]
            z_imag = z[:, dim_z:]
            z_complex = torch.complex(z_real, z_imag)
            power = (z_complex.real ** 2 + z_complex.imag ** 2).sum(dim=1, keepdim=True)
            z_in = z_complex * torch.complex(torch.sqrt(torch.tensor(dim_z, dtype=torch.float32, device=z.device) / power), torch.tensor(0.0, device=z.device))
            z_out_complex, h = fading(z_in, noise_stddev, prev_h)
            z_out = torch.cat([z_out_complex.real, z_out_complex.imag], dim=1)

        elif self.channel_type == "fading-real":
            dim_z = z.size(1) // 2
            z_in = torch.sqrt(torch.tensor(dim_z, dtype=torch.float32, device=z.device)) * F.normalize(z, p=2, dim=1)
            z_out, h = phase_invariant_fading(z_in, noise_stddev, prev_h)

        else:
            raise ValueError("Unknown channel type")

        z_out = z_out.view_as(encoded_img)
        avg_power = torch.mean(z_in ** 2)
        return z_out, avg_power, h




class DeepJSCCF(nn.Module):
    def __init__(self, channel_snr, conv_depth, channel_type, feedback_snr, refinement_layer, layer_id, target_analysis=False):
        super(DeepJSCCF, self).__init__()
        self.refinement_layer = refinement_layer
        self.feedback_snr = feedback_snr
        self.layer = layer_id
        self.encoder = Encoder(conv_depth)
        self.decoder = Decoder(128)
        self.channel = Channel(channel_type, channel_snr)
        if self.refinement_layer:
            self.image_combiner = OutputsCombiner()
        self.target_analysis = target_analysis

    def forward(self, inputs):
        if self.refinement_layer:
            img, prev_img_out_fb, prev_img_out_dec, prev_chn_out_fb, prev_chn_out_dec, prev_chn_gain = inputs
            img_in = torch.cat([prev_img_out_fb, img], dim=0)
        else:
            img = inputs
            img_in = inputs
            prev_chn_gain = None

        chn_in = self.encoder(img_in)
        chn_out, avg_power, chn_gain = self.channel(chn_in, prev_chn_gain)

        if self.feedback_snr is None:
            chn_out_fb = chn_out
        else:
            fb_noise_stddev = np.sqrt(10 ** (-self.feedback_snr / 10))
            chn_out_fb = real_awgn(chn_out, fb_noise_stddev)

        if self.refinement_layer:
            chn_out_exp = torch.cat([chn_out, prev_chn_out_dec], dim=0)
            residual_img = self.decoder(chn_out_exp)
            decoded_img = self.image_combiner(prev_img_out_dec, residual_img)

            chn_out_exp_fb = torch.cat([chn_out_fb, prev_chn_out_fb], dim=0)
            residual_img_fb = self.decoder(chn_out_exp_fb)
            decoded_img_fb = self.image_combiner(prev_img_out_fb, residual_img_fb)
        else:
            decoded_img = self.decoder(chn_out)
            decoded_img_fb = self.decoder(chn_out_fb)

        return decoded_img, decoded_img_fb, chn_out, chn_out_fb, chn_gain

    def change_channel_snr(self, channel_snr):
        self.channel.channel_snr = channel_snr

    def change_feedback_snr(self, feedback_snr):
        self.feedback_snr = feedback_snr








def train_and_evaluate(args):
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    prev_layer_out = None
    all_models = []

    for layer in range(args.n_layers):
        print(f"\n=== Training Layer {layer} ===")

        feedback_snr = None if not args.feedback_noise else args.feedback_snr_train
        channel_snr = args.channel_snr_train

        model = DeepJSCCF(
            channel_snr=channel_snr,
            conv_depth=args.conv_depth,
            channel_type=args.channel,
            feedback_snr=feedback_snr,
            refinement_layer=(layer > 0),
            layer_id=layer,
            target_analysis=args.target_analysis
        ).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
        criterion = torch.nn.MSELoss()
        logger = NBatchLogger(display=100)
        psnr_var = PSNRsVar()
        psnr_hist = TargetPSNRsHistogram() if args.target_analysis else None

        model.train()
        for epoch in range(args.train_epochs):
            total_loss = 0
            for i, (x, _) in enumerate(train_loader):
                x = x.to(args.device)
                optimizer.zero_grad()

                if layer == 0:
                    decoded_img, *_ = model(x)
                else:
                    decoded_img, *_ = model((x, *prev_layer_out))

                loss = criterion(decoded_img, x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                psnr_var.update(x, decoded_img)
                if psnr_hist:
                    psnr_hist.update(x, decoded_img)

                logger.log({'loss': loss.item()})

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, PSNR Var: {psnr_var.result():.4f}")
            psnr_var.reset()
            if psnr_hist:
                print(f"PSNR Histogram: {psnr_hist.result()}")
                psnr_hist.reset()

        # Prepare for next layer
        model.eval()
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(args.device)
                if layer == 0:
                    decoded_img, decoded_img_fb, chn_out, chn_out_fb, chn_gain = model(x)
                else:
                    decoded_img, decoded_img_fb, chn_out, chn_out_fb, chn_gain = model((x, *prev_layer_out))
                prev_layer_out = (decoded_img, decoded_img_fb, chn_out, chn_out_fb, chn_gain)
                break

        all_models.append(model)
        torch.save(model.state_dict(), f'C:/TransUnrecon/VAE_MNIST_2/comparisons/DeepJSCC_f_layer{layer}.pth')

    print("\n=== Evaluation ===")
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(args.device)
            decoded_img, *_ = model(x)
            mse = torch.nn.functional.mse_loss(decoded_img, x)
            total_mse += mse.item()

    avg_mse = total_mse / len(val_loader)
    print(f"Final Validation MSE: {avg_mse:.4f}")



class Args:
    data_dir = 'C:/TransUnrecon/data'
    model_dir = 'C:/TransUnrecon/VAE_MNIST_2/comparisons'
    batch_size = 100
    n_layers = 3
    conv_depth = 128
    channel = 'awgn'
    channel_snr_train = 15
    feedback_snr_train = 15
    feedback_noise = True
    train_epochs = 15
    learn_rate = 1e-3
    target_analysis = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    args = Args()
    train_and_evaluate(args)