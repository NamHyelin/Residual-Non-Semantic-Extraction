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
import re

from classifier import CNN
from Diffusion_model import SimpleUnet, denoise2, forward_diffusion_sample, sample_random_image, linear_beta_schedule
from ImageNet import ImageNet
from VAE_model import VAE
from unexpressed import calcul_A_ImageNet
from BER import Ax_error
from lpips_score import lpips, pfw, LPIPS, FID, KID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_semantic(bidx, method, save_method, classifier):
    # lpips_model = lpips.LPIPS(net='alex').to(device)
    with torch.no_grad():

        sem_losses = np.zeros((14))
        lpi_losses = np.zeros((14))
        fid_losses = np.zeros((14))
        cls_losses = np.zeros((14))

        for batch_idx, (x_, y) in enumerate(validation_loader):
            x = x.view(-1, 1, 28, 28)
            x = x.to(device)

            t = torch.randint(0, T, (batch_size,), device=device).long()

            mu, _ = encoder.encode(x.view(-1, x_dim))
            z = mu.view(encoder_dim)
            x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
            noise_pred = model(x_noisy, t)
            recon_x = denoise2(x_noisy, T, noise_pred)

            # semantic_loss = nn.CrossEntropyLoss()(classifier(recon_x.view(-1, 1, 28, 28)), y).item()
            # lpips_loss = 0
            # for l in range(batch_size):
            #     lpips_loss += lpips_model.forward(transforms.Resize((256,256))(x.view(100,1,28,28))[l].to(device), transforms.Resize((256,256))(recon_x.view(100,1,28,28))[l].to(device)).item()
            # FID_loss = pfw.fid(transforms.Resize((256,256))(x.view(100,1,28,28).expand(-1,3,-1,-1)).to(device), transforms.Resize((256,256))(recon_x.view(100,1,28,28).expand(-1,3,-1,-1)).to(device))/3
            class_error = (y.size(0) - (classifier(recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)

            # TODO plot under-expessed term
            sem_loss = []
            lpi_loss = []
            fid_loss = []
            cls_loss = []

            # sem_loss.append(semantic_loss/batch_size)
            # lpi_loss.append(lpips_loss/batch_size)
            # fid_loss.append(FID_loss/batch_size)
            cls_loss.append(class_error)

            for d in np.arange(2, 28, 2):
                if type(bidx) == str:
                    A = torch.zeros((d, training_data.data.shape[1])).to(device)
                    for c in range(int(bidx)):
                        A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                    A /= c
                else:
                    A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)

                Ax = torch.matmul(A, x.view(-1, 28, 28))
                A_ = torch.linalg.pinv(A)

                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                # semantic_loss = nn.CrossEntropyLoss()(classifier(final_recon_x.view(-1, 1, 28, 28)), y).item()
                # lpips_loss = 0
                # for l in range(batch_size):
                #     lpips_loss += lpips_model.forward(transforms.Resize((256, 256))(x.view(100, 1, 28, 28))[l].to(device), transforms.Resize((256, 256))(final_recon_x.view(100, 1, 28, 28))[l].to(device)).item()
                # FID_loss = pfw.fid(transforms.Resize((256, 256))(x.view(100, 1, 28, 28).expand(-1, 3, -1, -1)).to(device), transforms.Resize((256, 256))(final_recon_x.view(100, 1, 28, 28).expand(-1, 3, -1, -1)).to(device)) / 3
                class_error = (y.size(0) - (classifier(final_recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)

                # sem_loss.append(semantic_loss / batch_size)
                # lpi_loss.append(lpips_loss / batch_size)
                # fid_loss.append(FID_loss / batch_size)
                cls_loss.append(class_error)

            # sem_losses += sem_loss
            # lpi_losses += lpi_loss
            # fid_losses += fid_loss
            cls_losses += cls_loss

        # if type(bidx) == str:
        #     np.save(f'results/final_classify_loss_{method}_bidx_avg'+bidx+'.npy', sem_losses/batch_idx)
        # else:
        #     np.save(f'results/final_classify_loss_{method}_bidx_{bidx}.npy', sem_losses / batch_idx)
        #
        # if type(bidx) == str:
        #     np.save(f'results/final_lpips_loss_{method}_bidx_avg'+bidx+'.npy', lpi_losses/batch_idx)
        # else:
        #     np.save(f'results/final_lpips_loss_{method}_bidx_{bidx}.npy', lpi_losses / batch_idx)
        #
        # if type(bidx) == str:
        #     np.save(f'results/final_fid_loss_{method}_bidx_avg'+bidx+'.npy', fid_losses/batch_idx)
        # else:
        #     np.save(f'results/final_fid_loss_{method}_bidx_{bidx}.npy', fid_losses / batch_idx)

        if type(bidx) == str:
            np.save(f'results/final_classify_error_{save_method}_bidx_avg' + bidx + '.npy', cls_losses / batch_idx)
        else:
            np.save(f'results/final_classify_error_{save_method}_bidx_{bidx}.npy', cls_losses / batch_idx)

        plt.subplot(2, 2, 1)
        plt.plot(sem_losses / batch_idx, label='')
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('Classification CE loss')
        plt.title('Semantic error1 wrt d')

        plt.subplot(2, 2, 2)
        plt.plot(lpi_losses / batch_idx, label='')
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('LPIPS loss')
        plt.title('Semantic error2 wrt d')

        plt.subplot(2, 2, 3)
        plt.plot(fid_losses / batch_idx, label='')
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('FID loss')
        plt.title('Semantic error3 wrt d')

        plt.subplot(2, 2, 4)
        plt.plot(cls_losses / batch_idx, label='')
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('Classification error')
        plt.title('Semantic error4 wrt d')


def test(bidx, method, save_method):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses = np.zeros((14))
        for batch_idx, (x, y) in enumerate(training_loader):
            print(f'[{batch_idx}] batch idx')
            x = x.view(-1, 3, 224, 224)
            x = x.to(device)

            x_noisy_test, noise_test = forward_diffusion_sample(x.to(device), (T - 1) * torch.ones((batch_size)).to(device).long())
            recon_x = sample_random_image(noised_img=x_noisy_test.to(device), T=T, std=1, mean=0, model=model)
            reconst_loss = nn.MSELoss()(recon_x.view(-1, 3, 224, 224), x.view(-1, 3, 224, 224)).item()

            # TODO plot under-expessed term
            d_loss = []
            d_loss.append(reconst_loss / batch_size)
            for d in np.arange(2, 28, 2):
                if 'SecondOrder' in save_method:
                    if type(bidx) == str:
                        A = torch.zeros((d, training_data.data.shape[1])).to(device)
                        for c in range(int(bidx)):
                            A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                        A /= c
                    else:
                        A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)
                else:
                    A = calcul_A_ImageNet(x, recon_x, d, True).to(device)

                Ax = torch.matmul(A, x.view(-1, 3, 224, 224))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 3, 224, 224))) + torch.matmul((torch.eye(224).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 3, 224, 224))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 3, 224, 224)) + torch.matmul((torch.eye(224).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 3, 224, 224))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1, 3, 224, 224)).item()
                d_loss.append(final_reconst_loss / batch_size)
            d_losses += d_loss

            if 'SecondOrder' in save_method:
                if type(bidx) == str:
                    np.save(f'results/final_mse_loss_{save_method}_bidx_avg' + bidx + '.npy', d_losses / batch_idx)
                else:
                    np.save(f'results/final_mse_loss_{save_method}_bidx_{bidx}.npy', d_losses / batch_idx)
            else:
                np.save(f'results/final_mse_loss_{save_method}.npy', d_losses / batch_idx)

        print('original loss: ', origin_mse_loss / batch_idx, '\n')
        print('final losses wrt d: ', d_losses / batch_idx)
        plt.plot(d_losses / batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')
        plt.show()



def test_oneimg(bidx, method, save_method, d):
    if type(bidx) == str:
        A = torch.zeros((d, training_data.data.shape[1])).to(device)
        for c in range(int(bidx)):
            A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
        A /= c
    else:
        A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)

    with torch.no_grad():
        final_losses = []
        initial_losses = []
        for batch_idx, (x_, y) in enumerate(training_loader):
            for imgidx in range(batch_size):
                x = x_[imgidx].unsqueeze(0).view(1, x_dim)
                x = x.view(-1, 1, 28, 28)
                x = x.to(device)

                t = torch.randint(0, T, (batch_size,), device=device).long()
                mu, _ = encoder.encode(x.view(-1, x_dim))
                z = mu.view(encoder_dim)
                x_noisy, noise = forward_diffusion_sample(T, x, z, t, device)
                noise_pred = model(x_noisy, t)
                recon_x = denoise2(x_noisy, T, noise_pred)

                reconst_loss = nn.MSELoss()(recon_x.view(-1, 28, 28), x.view(-1, 28, 28)).item()

                # TODO plot under-expessed term
                Ax = torch.matmul(A, x.view(-1, 28, 28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1, 28, 28)).item()

                final_losses.append(final_reconst_loss)
                initial_losses.append(reconst_loss)

            np.save(f'results/final_mse_loss_{save_method}.npy', final_losses)
            np.save(f'results/initial_mse_loss_{save_method}.npy', initial_losses)




if __name__ == "__main__":
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
    batch_size = 50
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Parameters setting
    x_dim = 224 ** 2
    hidden_dim = 256
    learning_rate = 1e-3
    method = 'DDPM'
    model_method = 'DDPM'
    save_method = 'DDPM_batchsize50'
    bidx = 0

    # Load model
    down_channels = (32, 64, 128, 256)
    up_channels = (256, 128, 64, 32)
    model = SimpleUnet(down_channels, up_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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

    # data_variance = np.var(training_data.data / 255.0)  # 0.06328692405746414
    # test_data_variance = np.var(validation_data.data / 255.0)  # 0.06311123249672204


    # Load model
    down_channels = (32, 64, 128, 256)
    up_channels = (256, 128, 64, 32)
    model = SimpleUnet(down_channels, up_channels).to(device)
    try:
        model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + model_method + '.pth'))
    except:
        print('wrong model')
        model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + 'DDPM' + '.pth'))



    test(bidx=bidx, method=method, save_method=save_method)
    # test_oneimg(bidx, method, save_method, 18)

    plt.legend()
    plt.show()



    '''
    # plot TSNE
    tsne = TSNE(n_components=2, n_iter=1000, perplexity=2)
    digits_tsne = tsne.fit_transform(z.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
    for i in range(len(dbscan_labels)):
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], y[i].item())
    plt.show()


    # plot img
    plt.subplot(2,1,1)
    plt.imshow(x[0].detach().cpu().numpy().reshape(28, 28), cmap="gray")
    plt.title('Input data')
    plt.subplot(2,1,2)
    plt.imshow(recon_x[0].detach().cpu().numpy().reshape(28, 28), cmap="gray")
    plt.title('Reconstructed data')
    plt.show()


    plt.subplot(3, 1, 1)
    plt.imshow(A[0], cmap="gray")
    plt.title('calculated A')
    plt.subplot(3, 1, 2)
    plt.imshow(Ax[0], cmap="gray")
    plt.title('Transmitting under-expressed terms')
    plt.subplot(3, 1, 3)
    plt.imshow(final_recon_x[0], cmap="gray")
    plt.title('Final reconstructed data')
    plt.show()

    '''
