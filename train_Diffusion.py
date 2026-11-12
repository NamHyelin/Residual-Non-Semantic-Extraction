import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
import random
from classifier import CNN
from unexpressed import calcul_A_MNIST, calcul_A_ImageNet
from ImageNet import ImageNet
from VAE_model import VAE
from tqdm import tqdm
import time
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
https://velog.io/@rcchun/%EB%94%A5%EB%9F%AC%EB%8B%9DDiffusion-model-from-scratch-in-pytorch
https://medium.com/@mickael.boillaud/denoising-diffusion-model-from-scratch-using-pytorch-658805d293b4
https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffae/

Semantic communication
https://arxiv.org/pdf/2401.06803

'''


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    # mean + variance
    # latent DDIM 은 x_0 대신에 z가 들어감 #아님. 더 복잡
    # sqrt_alphas_cumprod_t 이런 스케줄링은 서로가 아는걸로.
    z = F.interpolate(x_0, size=x_0.shape[-2:], mode='bilinear', align_corners=False)
    noised_x = sqrt_alphas_cumprod_t.to(device) * z.to(device) \
               + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    return noised_x, noise



class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=3, padding=1)  # 31
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  # 421
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)  # 31
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # 421
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, down_channels, up_channels):
        super().__init__()
        image_channels = 3  # 3
        self.down_channels = down_channels
        self.up_channels = up_channels
        out_dim = 3  # 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(self.down_channels[i], self.down_channels[i + 1], time_emb_dim) \
                                    for i in range(len(self.down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(self.up_channels[i], self.up_channels[i + 1], time_emb_dim, up=True) \
                                  for i in range(len(self.up_channels) - 1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(self.up_channels[-1], out_dim, kernel_size=1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # if residual_x.size() != x.size():
            #     diffY = residual_x.size()[2] - x.size()[2]
            #     diffX = residual_x.size()[3] - x.size()[3]
            #     x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)








def denoise2(noisy_img, T, noise_pred):
    return noisy_img - noise_pred






@torch.no_grad()
def sample_random_image(noised_img='random', T=300, std=1, mean=0, model=SimpleUnet):
    # Sample noise
    if noised_img=='random':
        img = torch.randn((1, 3, 224, 224), device=device)*std + mean
    else:
        img = noised_img


    # for i in range(0,T)[::-1]:
    for i in tqdm(range(T-1, -1, -1), position=0, leave=True, desc='Diffusion step'):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """

        betas_t = get_index_from_list(betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, img.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, img.shape)

        # Call model (current image - noise prediction)
        # 매번 denoise 조금씩하고 그때마다 model 이 noise 를 predict 함.
        model_mean = sqrt_recip_alphas_t * (
                img - betas_t * model(img, t) / sqrt_one_minus_alphas_cumprod_t
        )
        # print(f'[{i}/{T}] denoising done')
        posterior_variance_t = get_index_from_list(posterior_variance, t, img.shape)

        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x).to(device)
        img = model_mean + torch.sqrt(posterior_variance_t) * noise

        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

    return img






if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # ImageNet = "https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    training_data = ImageNet("C:/TransUnrecon/data/ImageNet/imagenet-object-localization-challenge/", split="train", transform=transform)
    validation_data = ImageNet("C:/TransUnrecon/data/ImageNet/imagenet-object-localization-challenge/", split="val", transform=transform)
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
    batch_size = 20
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True)



    # Parameters setting
    num_epochs = 50  #50
    x_dim = 224**2
    hidden_dim = 256
    learning_rate = 1e-3
    method= 'DDPM'
    model_method = 'DDPM'
    load = True


    # Load model
    down_channels = (32, 64, 128, 256)
    up_channels = (256, 128, 64, 32)
    model = SimpleUnet(down_channels, up_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if load==True:
        model.load_state_dict(torch.load(f'C:/TransUnrecon/VAE_MNIST_2/{model_method}.pth'))


    # Define beta schedule
    T = 150
    betas = linear_beta_schedule(start=0.0001, end=0.02, timesteps=T)
    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)



    reconst_loss_train, noise_loss_train = [], []
    reconst_loss_test, noise_loss_test, semantic_loss_test = [], [], []

    for e in range(num_epochs):
        reconst_loss_tr, noise_loss_tr = 0.0, 0.0
        reconst_loss_ts, noise_loss_ts, semantic_loss_ts = 0.0, 0.0, 0.0

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            # print(f'Batch training [{batch_idx_tr} / {len(training_loader)}]')
            model.train()
            x = x.view(-1, 3, 224, 224)
            x = x.to(device)

            optimizer.zero_grad()
            t = torch.randint(0, T, (batch_size,), device=device).long()

            x_noisy, noise = forward_diffusion_sample(x, t)
            noise_pred = model(x_noisy, t)


            loss = nn.MSELoss()(noise, noise_pred)  #F.l1_loss
            noise_loss_tr += loss.item()


            loss.backward()
            optimizer.step()



            #TODO calculate A
            if e == num_epochs -1 :
                print('saving A for batch ', batch_idx_tr)
                for d in np.arange(2, 224, 11):
                    x_noisy_test, noise_test = forward_diffusion_sample(x.to(device), (T-1)*torch.ones((batch_size)).to(device).long())
                    recon_x = sample_random_image(noised_img=x_noisy_test.to(device), T=T, std=1, mean=0, model=model)
                    x, recon_x = x.cpu(), recon_x.cpu()
                    if '_A' in method:
                        # smalleigen
                        A = calcul_A_ImageNet(x, recon_x, d, False).cpu()
                    else:
                        A = calcul_A_ImageNet(x, recon_x, d, True).cpu()  # (d, 224)
                    try:
                        torch.save(A, f'As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')
                    except:
                        os.makedirs(f'As/{method}')
                        torch.save(A, f'As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')





        # with torch.no_grad():
        #     for batch_idx_ts, (x, label) in enumerate(validation_loader):
        #         if batch_idx_ts<20:
        #             x = x.view(batch_size, 3, 224, 224).to(device)
        #             label = label.to(device)
        #
        #             t = torch.randint(0, T, (batch_size,), device=device).long()
        #             x_noisy, noise = forward_diffusion_sample(x, t)
        #
        #             noise_pred = model(x_noisy, t)
        #             recon_x = sample_random_image(noised_img=x_noisy, T=300, std=1, mean=0, model=model)
        #
        #             reconst_loss = nn.MSELoss()(recon_x, x)
        #             noise_loss = nn.MSELoss()(noise, noise_pred)
        #
        #             reconst_loss_ts += reconst_loss.item()
        #             noise_loss_ts += noise_loss.item()


        reconst_loss_train.append(reconst_loss_tr / batch_idx_tr)
        noise_loss_train.append(noise_loss_tr / batch_idx_tr)
        # reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
        # noise_loss_test.append(noise_loss_ts / batch_idx_ts)

        print(f'[{e} / {num_epochs}] reconstruction loss: {reconst_loss_train[-1]}, noise loss: {noise_loss_train[-1]}', time.strftime('%X', time.localtime()))



        # np.save('results/reconst_loss_train_'+ method + '.npy', reconst_loss_train)
        # np.save('results/noise_loss_train_' + method + '.npy', noise_loss_train)
        # # np.save('results/reconst_loss_test_' + method + '.npy', reconst_loss_test)
        # torch.save(model.state_dict(), model_method+'.pth')