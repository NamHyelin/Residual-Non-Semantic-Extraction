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
from unexpressed import calcul_A_MNIST, calcul_A_eigenvalues
from QAM import snr, Ps_qam
from SNR import add_noise
from VAE_model import VAE
import sys
from scipy.sparse import csr_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def sym_CSR(matrix, threshold):
    upper_triangular = np.triu(torch.where(matrix.abs() < threshold, 0.0, matrix).detach().cpu().numpy())
    sparse_matrix = csr_matrix(upper_triangular)

    return sparse_matrix





def reconst_CSR(data, indices, indptr):
    n = len(indptr) - 1

    reconstructed_matrix = np.zeros((n, n))
    for row in range(n):
        start_index = indptr[row]
        end_index = indptr[row + 1]
        for idx in range(start_index, end_index):
            col = indices[idx]
            reconstructed_matrix[row, col] = data[idx]

    for i in range(n):
        for j in range(i + 1, n):
            reconstructed_matrix[j, i] = reconstructed_matrix[i, j]

    return reconstructed_matrix



def calcul_A_matrix(matrix, d):
    eigenvalues, eigenvectors = torch.linalg.eig(matrix.float())  # (28), #(28,28)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    # eigenvalues[eigenvalues.abs() < 1e-5] = 0

    _, indices = torch.topk(eigenvalues, k=d, dim=0, largest=True)
    D = 28
    s = eigenvalues.size()[0]
    A = torch.zeros((s, d))
    for k in range(d):
        index = indices[k]
        A[:, k] = eigenvectors[:, index]
    # A = A.repeat(D, 1, 1)
    # A = A.permute(0, 2, 1)
    A = A.transpose(0, 1)
    return A








def test_B(R, data, recon_data, d, threshold):
    #TODO transmit compressed B-R
    sub = data - recon_data
    sub_reshaped = sub.view(1, 28, 28)
    sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

    B = torch.bmm(sub_reshaped, sub_T_reshaped).squeeze(0)
    c_BR = sym_CSR((B-R), threshold)


    #TODO receive and reconstruct
    BR = reconst_CSR(c_BR.data, c_BR.indices, c_BR.indptr)
    BR = torch.tensor(BR).to(device)
    B_rec = BR + R
    A_B = calcul_A_matrix(B_rec, d).to(device)

    A_Bx = torch.matmul(A_B, data.view(1, 28, 28))
    A_B_ = torch.linalg.pinv(A_B)
    final_recon_x_B = torch.matmul(torch.matmul(A_B_, A_B), data.view(1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_B_, A_B)), recon_data.view(1, 28, 28))
    final_reconst_loss_B = nn.MSELoss()(final_recon_x_B, data.view(1, 28, 28)).item()

    return c_BR.data.shape[0]+c_BR.indices.shape[0]+c_BR.indptr.shape[0], final_reconst_loss_B





if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()])) #, transforms.Normalize((0.1307), (0.3081))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()])) #, transforms.Normalize((0.1307), (0.3081))
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
    x_dim = training_data.data.shape[1]**2
    hidden_dim = 128
    learning_rate = 1e-3
    model_method = 'VAE'
    save_method = f'VAE_sparsematrix_drank_N_inclusive_th7'
    d = 18
    threshold = 0.07
    Ns= [1, 2, 5, 10, 20, 50, 100]
    img_idx = 0




    # Load model
    model = VAE(x_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/'+ model_method +'.pth'))





    for batch_idx, (x_, y) in enumerate(training_loader):
        x = x_.view(batch_size, x_dim)
        x = x.view(-1, x_dim)
        x = x.to(device)

        mu, logvar, z, recon_x = model(x)
        reconst_loss = nn.MSELoss()(recon_x.view(-1, 28, 28), x.view(-1, 28,28)).item()

        # TODO calculate R
        sub = x - recon_x
        sub_reshaped = sub.view(-1, 28, 28)
        sub_T_reshaped = sub_reshaped.permute(0, 2, 1)
        R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
        R = torch.mean(R_reshaped, 0)
        A = calcul_A_matrix(R, d).to(device)
        if batch_idx==0:
            break




    #AAA Test
    ds = []
    loss_recons = []
    for batch_idx, (x_, y) in enumerate(training_loader):
        x = x_.view(batch_size, x_dim)
        x = x.view(-1, x_dim)
        x = x.to(device)
        mu, logvar, z, recon_x = model(x)
        reconst_loss = nn.MSELoss()(recon_x.view(-1, 28, 28), x.view(-1, 28, 28)).item()
        #TODO Reconstruct with R
        Ax = torch.matmul(A, x.view(-1, 28, 28))
        A_ = torch.linalg.pinv(A)
        final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
        final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1, 28, 28)).item()

        for img_idx in range(batch_size):
            data = x[img_idx]
            recon_data = recon_x[img_idx]
            d = calcul_A_eigenvalues(data, recon_data).item()
            reconst_loss = nn.MSELoss()(recon_data.view(-1, 28, 28), data.view(-1, 28, 28)).item()
            ds.append(d)
            loss_recons.append(reconst_loss/100)

        np.save(f'results/d_foreachimginonebatch.npy', ds)
        np.save(f'results/mseloss_foreachimginonebatch.npy', loss_recons)
        print(batch_idx)
        if batch_idx==100:
            sys.exit()




'''


    #AAA test 2
    loss_ths = np.zeros((len(Ns)))
    comm_ths = np.zeros((len(Ns)))
    c_Ns = np.zeros((len(Ns)))
    for N_idx in range(len(Ns)):
        for batch_idx, (x_, y) in enumerate(training_loader):
            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar, z, recon_x = model(x)
            reconst_loss = nn.MSELoss()(recon_x.view(-1, 28, 28), x.view(-1, 28, 28)).item()

            #TODO calculate R
            sub = x[:Ns[N_idx]] - recon_x[:Ns[N_idx]]
            sub_reshaped = sub.view(-1, 28, 28)
            sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

            R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
            R = torch.mean(R_reshaped, 0)

            if N_idx==0:
                R=torch.zeros((28,28)).to(device)

            A = calcul_A_matrix(R, d).to(device)


            # TODO calculate B
            for img_idx in range(Ns[N_idx]):
                data = x[img_idx]
                recon_data = recon_x[img_idx]

                d = calcul_A_eigenvalues(data, recon_data).item()
                comm_load, final_reconst_loss_B = test_B(R, data, recon_data, d, threshold)
                # print('threshold: ', thresholds[thr_idx])
                # print('comm load: ', comm_load)
                # print('\n')
                loss_ths[N_idx] += final_reconst_loss_B
                comm_ths[N_idx] += comm_load
                c_Ns[N_idx]+=1

                print(f'[{batch_idx}/600], img idx {img_idx}/{Ns[N_idx]}')






            np.save(f'results/final_mse_loss_{save_method}.npy', loss_ths)
            np.save(f'results/comm_load_{save_method}.npy', comm_ths)
            np.save(f'results/Ns_{save_method}.npy', np.array([0, 2, 5, 10, 20, 50, 100]))
            np.save(f'results/cNs_{save_method}.npy', c_Ns)

'''

