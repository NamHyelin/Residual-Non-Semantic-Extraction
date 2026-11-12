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
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import time
from torch.utils.tensorboard import SummaryWriter
from VAE_model import VAE
from unexpressed import calcul_A_MNIST
from classifier import CNN

'''
2 layer씩 기본이고
test 시 unexpressed term 보냄
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TODO Vector quantizer layer

# 채널 단위로, 한 채널 단위당 나머지 dim들은 flatten 되어서 quantize 됨
# output은 input 과 같은 크기
# input [16, 64, 32, 32] 면 64 빼고 batchsize 도 다 flatten [16384, 64]
# 그럼 각 16384개의 64sized vector로 quantized.
# 전체 num_embeddings 중에 16384개 고르는
#AAA embedding dm?

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) # 업데이트되라고 codebook matrix 을 weight 으로 구성
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)  # weight init # (512, 64)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # (batchsize * latent_dim) 의 vae_enc output이 옴
        # (batchsize * latent_dim) / embedding_dim  개수의 latent 를 codebook 에서 골라 (embedding_dim 작을수록 좋을듯)
        # VAE 보다 latent_dim / embedding_dim 의 이득이 있음

        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape # (100, 256)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim) # (400, 64)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) # 원소별 제곱, sum 하면 (400,1)
                     + torch.sum(self._embedding.weight ** 2, dim=1) # (512)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t())) # (400, 512)
                    # ||x' - e.weight||^2  # (400, 512) # 각 400개가 512개 codebook 과의 distance 를 다 잼

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (400) # 이거 각각 512개 중에서 가장 작은걸 뽑음
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # encoding_indices 가 16384size 벡터고 각각이 512개중에 가장 가까운 index 저장하는데
                                                   # encodings 는 [0,첫번째index], [1,두번째index], [2,세번째index] 이렇게가 1임 나머지는 0
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) # 걍 quantize 결과 내놓음. 512개 각64size codebook 중 고르려고 나머지 0만들려고 곱한거

        # Loss 두개 다 update 되게.
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss  # commitment_cost는 codebook 이 너무 커서 update 가 느릴까봐 e_loss 의 비율을 낮춰준거

        quantized = inputs + (quantized - inputs).detach()  ###???????????????
        avg_probs = torch.mean(encodings, dim=0) # 얼마나 골고루 배정이 되는지 prob. 한 차원만 1개수 세어서 평균낸거임
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # entropy 가 sum(aloga) 인데 a가 어떤 일이 일어날 확률일때. 그걸 exponential 한듯

        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), perplexity, encodings






# TODO model
class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()

        self.vae = VAE(input_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        z, _ = self.vae.encode(x)
        loss, quantized, perplexity, _ = self.vq(z)
        x_recon = self.vae.decode(quantized)

        return loss, x_recon, perplexity





def train_1(method, x_dim, input_dim, num_embeddings, embedding_dim, commitment_cost, training_loader, validation_loader, num_epochs, lr, classifier):
    mse_losses, sem_losses = np.zeros((14)), np.zeros((14))
    for d in np.arange(2, 28, 2):
        c = int((128 + 28 * d) * 64)
        print('Hidden dim: ', c)
        model = VQVAE(input_dim, c, num_embeddings, embedding_dim, commitment_cost).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for e in range(num_epochs):
            print(f'[{e}/{num_epochs}]')
            for batch_idx_tr, (x, _) in enumerate(training_loader):
                model.train()
                x = x.view(-1, x_dim)
                x = x.to(device)
                optimizer.zero_grad()
                vq_loss, recon_x, _ = model(x)
                mse_loss = nn.MSELoss()(recon_x, x)
                loss = vq_loss * 100 + mse_loss
                loss.backward()
                optimizer.step()

        mse_loss, sem_loss = 0.0, 0.0
        for batch_idx_ts, (x, y) in enumerate(validation_loader):
            x = x.view(-1, x_dim)
            x, y = x.to(device), y.to(device)
            vq_loss, recon_x, _ = model(x)
            mse_loss += nn.MSELoss()(recon_x, x).item() / batch_size
            sem_loss += (y.size(0) - (classifier(recon_x.view(-1,1,28,28)).argmax(dim=1) == y).int().sum().item()) / batch_size

        mse_losses[int(d / 2) - 1] = mse_loss / batch_idx_ts
        sem_losses[int(d / 2) - 1] = sem_loss / batch_idx_ts

        np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{method}.npy', mse_losses)
        np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{method}.npy', sem_losses)















def train_2(method, x_dim, input_dim, latent_dim, num_embeddings, embedding_dim, commitment_cost, bidx, training_loader, validation_loader, num_epochs, lr, classifier):
    model = VQVAE(input_dim, latent_dim, num_embeddings, embedding_dim, commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reconst_loss_test, semantic_loss_test = [], []
    for e in range(num_epochs):
        reconst_loss_ts, semantic_loss_ts = 0.0, 0.0

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            model.train()
            x = x.view(-1, x_dim)
            x = x.to(device)
            optimizer.zero_grad()
            vq_loss, recon_x, _ = model(x)
            mse_loss = nn.MSELoss()(recon_x, x)#/batch_size
            loss = vq_loss*100  + mse_loss
            loss.backward()
            optimizer.step()

            # TODO calculate A
            if e == num_epochs - 1:
                print('saving A for batch ', batch_idx_tr)
                for d in np.arange(2, 28, 2):
                    x, recon_x = x.cpu(), recon_x.cpu()
                    A = calcul_A_MNIST(x, recon_x, d, True).cpu()  # (d, 28)
                    try:
                        torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')
                    except:
                        os.makedirs(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}')
                        torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')
        if e == int(num_epochs / 2):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / 10

        with torch.no_grad():
            for batch_idx_ts, (x, label) in enumerate(validation_loader):
                x = x.view(-1, x_dim)
                x, label = x.to(device), label.to(device)
                _, recon_x, _ = model(x)

                reconst_loss = nn.MSELoss()(recon_x, x)
                classifier_loss = nn.CrossEntropyLoss()(classifier(recon_x.view(-1, 1,28,28)), label)

                reconst_loss_ts += reconst_loss.item() / batch_size
                semantic_loss_ts += classifier_loss.item() / batch_size

        reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
        semantic_loss_test.append(semantic_loss_ts / batch_idx_ts)

        print(f'[{e} / {num_epochs}] reconstruction loss: {reconst_loss_test[-1]}, semantic loss: {semantic_loss_test[-1]}')

        np.save('C:/TransUnrecon/VAE_MNIST_2/results/comparisons/reconst_loss_test_' + method + '.npy', reconst_loss_test)
        np.save('C:/TransUnrecon/VAE_MNIST_2/results/comparisons/semantic_loss_test_' + method + '.npy', semantic_loss_test)
        torch.save(model.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/' + method + '.pth')



    print('\ntest with A')
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses, c_errors = np.zeros((14)), np.zeros((14))
        for batch_idx, (x, y) in enumerate(validation_loader):
            x = x.view(-1, x_dim)
            x = x.to(device)

            _, recon_x, _ = model(x)
            reconst_loss = nn.MSELoss()(recon_x, x).item()
            classify_error = nn.CrossEntropyLoss()(classifier(recon_x.view(-1, 1,28,28)), label).item()

            #TODO plot under-expessed term
            d_loss, c_error = [], []
            d_loss.append(reconst_loss/batch_size)
            c_error.append(classify_error / batch_size)
            for d in np.arange(2, 28, 2):
                if type(bidx)==str:
                    A = torch.zeros((d,training_data.data.shape[1])).to(device)
                    for j in range(int(bidx)):
                        A += torch.load(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{j}.pt').to(device)
                    A /= j
                else:
                    A = torch.load(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{bidx}.pt').to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                final_recon_x = torch.matmul(torch.matmul(A_,A), x.view(-1,28,28)) + torch.matmul((torch.eye(28).to(device)-torch.matmul(A_,A)) ,recon_x.view(-1,28,28))

                final_reconst_loss = nn.MSELoss()(final_recon_x.view(-1,28,28), x.view(-1,28,28)).item()
                final_classify_error = nn.CrossEntropyLoss()(classifier(final_recon_x.view(-1,1,28,28)), label).item()
                d_loss.append(final_reconst_loss/batch_size)
                c_error.append(final_classify_error/batch_size)
            d_losses += d_loss
            c_errors += c_error


        if type(bidx) == str:
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{method}_bidx_avg'+bidx+'.npy', d_losses/batch_idx)
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{method}_bidx_avg' + bidx + '.npy', d_losses / batch_idx)
        else:
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_mse_loss_{method}_bidx_{bidx}.npy', d_losses / batch_idx)
            np.save(f'C:/TransUnrecon/VAE_MNIST_2/results/comparisons/final_classify_error_{method}_bidx_{bidx}.npy', d_losses / batch_idx)
        print('original loss: ', origin_mse_loss/batch_idx, '\n')
        print('final losses wrt d: ', d_losses/batch_idx)
        plt.subplot(1,2,1)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')
        plt.subplot(1, 2, 2)
        plt.plot(c_errors / batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('Classify error')
        plt.title('Semantic error wrt d')
        plt.show()




if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))


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
    num_epochs = 60
    x_dim = training_data.data.shape[1]**2
    input_dim = training_data.data.shape[1] ** 2
    hidden_dim, latent_dim = 128*64, 128*64  # VAE encoder output # 128*2*64 였어야되는데......
    learning_rate = 1e-3

    hidden_ch = 32
    embedding_dim = 64 # output의 channel 개수. quantize 해야되는 vector 의 개수. latent_dim/embedding_dim 이 보내는차원
    num_embeddings = 512 # num_codebooks
    commitment_cost = 0.25

    bidx = 0
    lr = 1e-3
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))
    method = 'VQVAE_1'

    train_1(method, x_dim, input_dim, num_embeddings, embedding_dim, commitment_cost, training_loader, validation_loader, num_epochs, lr, classifier)
    # train_2(method, x_dim, input_dim, latent_dim, num_embeddings, embedding_dim, commitment_cost, bidx, training_loader, validation_loader, num_epochs, lr, classifier)