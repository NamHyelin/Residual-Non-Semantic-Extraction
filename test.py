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
from train_VAE import VAE, get_gauss, train_loss, Tsne_idx
from unexpressed import calcul_A_MNIST
from BER import Ax_error
from lpips_score import lpips, pfw, LPIPS, FID, KID, LPIPS_MNIST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def test_semantic(bidx, method, save_method, classifier):
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    with torch.no_grad():

        sem_losses=np.zeros((14))
        lpi_losses = np.zeros((14))
        fid_losses = np.zeros((14))
        cls_losses = np.zeros((14))

        for batch_idx, (x_, y) in enumerate(validation_loader):

            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x, y = x.to(device), y.to(device)

            mu, logvar, z, recon_x = model(x)
            # semantic_loss = nn.CrossEntropyLoss()(classifier(recon_x.view(-1, 1, 28, 28)), y).item()
            lpips_loss = LPIPS_MNIST(lpips_model, x.view(-1, 1, 28, 28), recon_x.view(-1, 1, 28, 28)).item()
            # FID_loss = pfw.fid(transforms.Resize((256,256))(x.view(100,1,28,28).expand(-1,3,-1,-1)).to(device), transforms.Resize((256,256))(recon_x.view(100,1,28,28).expand(-1,3,-1,-1)).to(device))/3
            # class_error = (y.size(0) - (classifier(recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)


            #TODO plot under-expessed term
            sem_loss = []
            lpi_loss = []
            fid_loss = []
            cls_loss = []

            # sem_loss.append(semantic_loss/batch_size)
            lpi_loss.append(lpips_loss/batch_size)
            # fid_loss.append(FID_loss/batch_size)
            # cls_loss.append(class_error)

            for d in np.arange(2, 28, 2):
                if 'SecondOrder' in save_method:
                    if type(bidx)==str:
                        A = torch.zeros((d,training_data.data.shape[1])).to(device)
                        for c in range(int(bidx)):
                            A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                        A /= c
                    else:
                        A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)
                elif 'random' in save_method:
                    random_matrix = torch.randn(d, 28)
                    U, _, V = torch.linalg.svd(random_matrix, full_matrices=False)
                    scaled_U = U * (1 / torch.max(torch.abs(U)))
                    if scaled_U.size(1) < 28:
                        padding = torch.zeros((d, 28 - scaled_U.size(1)))
                        scaled_U = torch.cat([scaled_U, padding], dim=1)
                    A= scaled_U.to(device)
                else:
                    A = calcul_A_MNIST(x, recon_x, d, True).to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)

                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1,28,28))) + torch.matmul((torch.eye(28).to(device)-torch.matmul(A_,A)) ,recon_x.view(-1,28,28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_,A), x.view(-1,28,28)) + torch.matmul((torch.eye(28).to(device)-torch.matmul(A_,A)) ,recon_x.view(-1,28,28))

                # semantic_loss = nn.CrossEntropyLoss()(classifier(final_recon_x.view(-1, 1, 28, 28)), y).item()
                lpips_loss = LPIPS_MNIST(lpips_model, x.view(-1, 1, 28, 28), final_recon_x.view(-1, 1, 28, 28)).item()
                # FID_loss = pfw.fid(transforms.Resize((256, 256))(x.view(100, 1, 28, 28).expand(-1, 3, -1, -1)).to(device), transforms.Resize((256, 256))(final_recon_x.view(100, 1, 28, 28).expand(-1, 3, -1, -1)).to(device)) / 3
                # class_error = (y.size(0) - (classifier(final_recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)


                # sem_loss.append(semantic_loss / batch_size)
                lpi_loss.append(lpips_loss / batch_size)
                # fid_loss.append(FID_loss / batch_size)
                # cls_loss.append(class_error)

            # sem_losses += sem_loss
            lpi_losses += lpi_loss
            # fid_losses += fid_loss
            # cls_losses += cls_loss


        # if type(bidx) == str:
        #     np.save(f'results/final_classify_loss_{method}_bidx_avg'+bidx+'.npy', sem_losses/batch_idx)
        # else:
        #     np.save(f'results/final_classify_loss_{method}_bidx_{bidx}.npy', sem_losses / batch_idx)
        #
        if 'SecondOrder' in save_method:
            if type(bidx) == str:
                np.save(f'results/final_lpips_loss_{save_method}_bidx_avg'+bidx+'.npy', lpi_losses/batch_idx)
            else:
                np.save(f'results/final_lpips_loss_{save_method}_bidx_{bidx}.npy', lpi_losses / batch_idx)
        else:
            np.save(f'results/final_lpips_loss_{save_method}.npy', lpi_losses / batch_idx)
        #
        # if type(bidx) == str:
        #     np.save(f'results/final_fid_loss_{method}_bidx_avg'+bidx+'.npy', fid_losses/batch_idx)
        # else:
        #     np.save(f'results/final_fid_loss_{method}_bidx_{bidx}.npy', fid_losses / batch_idx)

        # if type(bidx) == str:
        #     np.save(f'results/final_classify_error_{save_method}_bidx_avg'+bidx+'.npy', cls_losses/batch_idx)
        # else:
        #     np.save(f'results/final_classify_error_{save_method}_bidx_{bidx}.npy', cls_losses / batch_idx)



        # plt.subplot(2,2,1)
        # plt.plot(sem_losses/batch_idx, label='')
        # plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        # plt.xlabel('d')
        # plt.ylabel('Classification CE loss')
        # plt.title('Semantic error1 wrt d')
        #
        # plt.subplot(2, 2, 2)
        # plt.plot(lpi_losses / batch_idx, label='')
        # plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        # plt.xlabel('d')
        # plt.ylabel('LPIPS loss')
        # plt.title('Semantic error2 wrt d')
        print('LPIPS: ', lpi_losses / batch_idx)
        #
        # plt.subplot(2, 2, 3)
        # plt.plot(fid_losses / batch_idx, label='')
        # plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        # plt.xlabel('d')
        # plt.ylabel('FID loss')
        # plt.title('Semantic error3 wrt d')
        #
        # plt.subplot(2, 2, 4)
        # plt.plot(cls_losses / batch_idx, label='')
        # plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        # plt.xlabel('d')
        # plt.ylabel('Classification error')
        # plt.title('Semantic error4 wrt d')









def test(bidx, method, save_method):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses=np.zeros((14))
        # x_xhat_covar=np.zeros((int(10000/batch_size), 14, batch_size, 28,28))
        # A_save = np.zeros((int(10000/batch_size), 14, 28,28))
        # save_losses = np.zeros((int(10000/batch_size), 14))
        c=0
        for batch_idx, (x_, y) in enumerate(validation_loader):
            print(f'batch: [{batch_idx}/{int(10000/batch_size)}]')
            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar, z, recon_x = model(x)
            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()


            #TODO plot under-expessed term
            d_loss = []
            d_loss.append(reconst_loss/batch_size)
            for d in np.arange(2, 28, 2):
                if 'SecondOrder' in save_method:
                    if type(bidx)==str:
                        A = torch.zeros((d,training_data.data.shape[1])).to(device)
                        for c in range(int(bidx)):
                            A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                        A /= c
                    else:
                        A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)
                elif 'random' in save_method:
                    random_matrix = torch.randn(d, 28)
                    U, _, V = torch.linalg.svd(random_matrix, full_matrices=False)
                    scaled_U = U * (1 / torch.max(torch.abs(U)))
                    if scaled_U.size(1) < 28:
                        padding = torch.zeros((d, 28 - scaled_U.size(1)))
                        scaled_U = torch.cat([scaled_U, padding], dim=1)
                    A= scaled_U.to(device)

                    # base_column = torch.rand(28) * 2 - 1  # 28 values in range [-1, 1]
                    # columns = [base_column]
                    # for cc in range(1, d):
                    #     if cc % 3 == 0:
                    #         new_column = torch.sin(base_column * (cc + 1) * 0.5)
                    #     elif cc % 3 == 1:
                    #         new_column = torch.cos(base_column * (cc + 1) * 0.3)
                    #     else:
                    #         new_column = torch.pow(base_column, (cc % 5) + 2) * torch.sign(base_column)
                    #     new_column = torch.clamp(new_column, -1, 1)
                    #
                    #     columns.append(new_column)
                    # A = torch.stack(columns, dim=0).to(device)
                else:
                    A = calcul_A_MNIST(x, recon_x, d,  True).to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()
                d_loss.append(final_reconst_loss/batch_size)

                # sub = x - recon_x
                # sub_reshaped = sub.view(-1, 28, 28)
                # sub_T_reshaped = sub_reshaped.permute(0, 2, 1)
                # R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
                # x_xhat_covar[batch_idx, int(d/2)-1]= R_reshaped.detach().cpu().numpy()
                # A_save[batch_idx, int(d/2)-1][:d]= A.detach().cpu().numpy()
                # save_losses[batch_idx, int(d/2)-1] = final_reconst_loss/batch_size


            d_losses += d_loss

            if batch_size*batch_idx > 1000:
                break

        if 'SecondOrder' in save_method:
            if type(bidx) == str:
                np.save(f'results/final_mse_loss_{save_method}_bidx_avg' + bidx + '.npy', d_losses / batch_idx)
            else:
                np.save(f'results/final_mse_loss_{save_method}_bidx_{bidx}.npy', d_losses / batch_idx)
            # np.save(f'results/x_xhat_covar_SecondOrder.npy', x_xhat_covar)
            # np.save('results/A_save_SecondOrder.npy', A_save)
            # np.save('results/all_final_mse_loss_SecondOrder.npy', save_losses)
        else:
            np.save(f'results/final_mse_loss_{save_method}.npy', d_losses / batch_idx)

        print('original loss: ', origin_mse_loss/batch_idx, '\n')
        print('final losses wrt d: ', d_losses/batch_idx)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')








def test_SNR(bidx, method, save_method):
    print('bidx should be smaller than 2000')
    snr_db = np.arange(0, 40, 4)
    ds = np.arange(2, 28, 2)
    d_losses = np.zeros((len(snr_db), len(snr_db), len(ds)+1))  # SNR_test, SNR_train, d (0~)
    with torch.no_grad():

        #TODO A
        As = torch.zeros((len(snr_db),len(ds),28,28)).to(device)
        snr_train_count = np.zeros((len(snr_db), len(ds)))
        snr_test_count = np.zeros(len(snr_db))
        folder_path = f'C:/TransUnrecon/VAE_MNIST_2/As/{method}'
        pattern = r'\d+'
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            snr_train = int(re.findall(pattern, file_name)[-1])
            d = int(re.findall(pattern, file_name)[0])
            if snr_train_count[int(snr_train/4), int(d/2)-1] < bidx: # bidx개씩 만큼의 snr sample을 평균치자
                As[int(snr_train/4), int(d/2)-1] += torch.cat((torch.load(f'As/{method}/{file_name}'), torch.zeros(28-d, 28)), dim=0).to(device)
        As /= bidx

        print('finished making A')

        for batch_idx, (x_, y) in enumerate(training_loader):
            print(f'batch index: [{batch_idx}/600]')
            x = x_.view(-1, x_dim)
            x = x.to(device)

            mu, logvar = model.encode(x)

            # TODO noise add to mu, logvar
            rand_idx = np.random.randint(len(snr_db))
            snr_test = snr_db[rand_idx]
            snr_test_count[int(snr_test/4)]+=1
            cap = 720 * np.log2(1 + snr_test) / 32  # 32bit로 나눠야 dim
            block_num = int((hidden_dim * 2 - cap) / 2)

            # blocked
            random_indices = np.random.choice(hidden_dim, size=block_num, replace=False)
            mu[:, random_indices] = 0
            logvar[:, random_indices] = 0
            # # reconst
            # for ri in list(random_indices):
            #     left_idx = next((j for j in range(ri - 1, -1, -1) if mu[0, j] != 0), 0)
            #     right_idx = next((j for j in range(ri + 1, len(logvar[0])) if mu[0, j] != 0), left_idx)
            #     mu[:, ri] = (mu[:, left_idx] + mu[:, right_idx]) / 2
            #     logvar[:, ri] = (logvar[:, left_idx] + logvar[:, right_idx]) / 2

            z = model.reparameterize(mu, logvar)
            recon_x = model.decode(z)

            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()


            #TODO plot under-expessed term
            d_losses[int(snr_test/4), :, 0] = reconst_loss/batch_size


            for snr_train_0 in range(len(snr_db)):
                for d_0 in range(len(ds)):
                    A = As[snr_train_0, d_0][:(d_0+1)*2,:]

                    Ax = torch.matmul(A, x.view(-1,28,28))
                    A_ = torch.linalg.pinv(A)
                    if 'Ax' in save_method:
                        final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                    else:
                        final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                    final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()

                    d_losses[int(snr_test/4), snr_train_0, d_0+1] += final_reconst_loss/batch_size


        for j in range(len(snr_db)):
            d_losses[j] /= snr_test_count[j]
        np.save(f'results/final_mse_loss_{save_method}_bidx_{bidx}.npy', d_losses)
        np.save(f'results/snr_test_count_{save_method}_bidx_{bidx}.npy', snr_test_count)






def test_SNR_A(bidx, method, save_method):
    print('bidx should be smaller than 2000')
    snr_db = np.arange(0, 40, 4)
    ds = np.arange(2, 28, 2)
    d_losses = np.zeros((len(snr_db), len(ds)+1))  # SNR_test, SNR_train, d (0~)
    snr_test_count = np.zeros(len(snr_db))
    with torch.no_grad():

        for batch_idx, (x_, y) in enumerate(training_loader):
            print(f'batch index: [{batch_idx}/600]')
            x = x_.view(-1, x_dim)
            x = x.to(device)

            mu, logvar = model.encode(x)

            # TODO noise add to mu, logvar
            rand_idx = np.random.randint(len(snr_db))
            snr_test = snr_db[rand_idx]
            snr_test_count[int(snr_test/4)]+=1
            cap = 720 * np.log2(1 + snr_test) / 32  # 32bit로 나눠야 dim
            block_num = int((hidden_dim * 2 - cap) / 2)

            # blocked
            random_indices = np.random.choice(hidden_dim, size=block_num, replace=False)
            mu[:, random_indices] = 0
            logvar[:, random_indices] = 0
            # # reconst
            # for ri in list(random_indices):
            #     left_idx = next((j for j in range(ri - 1, -1, -1) if mu[0, j] != 0), 0)
            #     right_idx = next((j for j in range(ri + 1, len(logvar[0])) if mu[0, j] != 0), left_idx)
            #     mu[:, ri] = (mu[:, left_idx] + mu[:, right_idx]) / 2
            #     logvar[:, ri] = (logvar[:, left_idx] + logvar[:, right_idx]) / 2

            z = model.reparameterize(mu, logvar)
            recon_x = model.decode(z)

            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()

            #TODO plot under-expessed term
            d_losses[int(snr_test/4), 0] += reconst_loss/batch_size
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

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1, 28, 28)).item()
                d_losses[int(snr_test/4), int(d/2)] += final_reconst_loss / batch_size


        for j in range(len(snr_db)):
            d_losses[j] /= snr_test_count[j]
        np.save(f'results/final_mse_loss_{save_method}_bidx_{bidx}.npy', d_losses)
        np.save(f'results/snr_test_count_{save_method}_bidx_{bidx}.npy', snr_test_count)







def test_SNR_2(bidx, method, save_method):
    #이게 찐
    from QAM import Ps_qam, snr
    from SNR_VAE import add_noise_qam, add_noise_gau
    d_losses = np.zeros((len(snr)+1, len(np.arange(2, 28, 2))+1))
    cls_losses = np.zeros((len(snr) + 1, len(np.arange(2, 28, 2)) + 1))
    with torch.no_grad():

        for batch_idx, (x_, y) in enumerate(validation_loader):
            print(f'Batch idx: [{batch_idx}/100]')
            x = x_.view(-1, x_dim)
            x = x.to(device)
            y = y.to(device)

            mu, logvar = model.encode(x)

            # TODO noise add to mu, logvar + Ax
            for avg in range(10):
                d_losses_avg = np.zeros((len(snr)+1, len(np.arange(2, 28, 2))+1))
                for ber_index in np.arange(1,len(snr)+1):
                    # mu, logvar = add_noise_gau(mu, ber_index - 1), add_noise_gau(logvar, ber_index - 1)

                    z = model.reparameterize(mu, logvar)
                    z = add_noise_gau(z, ber_index-1)
                    recon_x = model.decode(z)

                    reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
                    class_error = (y.size(0) - (classifier(recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)
                    d_losses_avg[ber_index-1, 0] += reconst_loss/batch_size
                    cls_losses[ber_index-1, 0] += class_error

                    #TODO plot under-expessed term
                    for d in np.arange(2, 28, 2):
                        if 'SecondOrder' in save_method:
                            if type(bidx) == str:
                                A = torch.zeros((d, training_data.data.shape[1])).to(device)
                                for c in range(int(bidx)):
                                    A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                                A /= c
                            else:
                                A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)
                        else: #instan
                            A = calcul_A_MNIST(x, recon_x, d, True).to(device)

                        Ax = torch.matmul(A, x.view(-1, 28, 28))
                        # Ax = (add_noise_gau(Ax, ber_index - 1)+Ax+Ax+Ax)/4
                        # if 'SecondOrder' not in save_method: #instan
                        #     A = add_noise_gau(A, ber_index - 1)
                        A_ = torch.linalg.pinv(A)

                        final_recon_x = torch.matmul(A_, Ax) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))


                        reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1, 28, 28)).item()
                        class_error = (y.size(0) - (classifier(final_recon_x.view(-1, 1, 28, 28)).argmax(dim=1) == y).int().sum().item()) / y.size(0)
                        d_losses_avg[ber_index - 1, int(d/2)] += reconst_loss / batch_size
                        cls_losses[ber_index - 1, int(d/2)] += class_error
                d_losses += d_losses_avg/10


        if 'SecondOrder' in save_method:
            np.save(f'results/final_mse_loss_{save_method}_bidx_{bidx}.npy', d_losses/batch_idx)
            np.save(f'results/final_classify_error_{save_method}_bidx_{bidx}.npy', cls_losses/batch_idx)
        else: #instan
            np.save(f'results/final_mse_loss_{save_method}.npy', d_losses / batch_idx)
            np.save(f'results/final_classify_error_{save_method}.npy', cls_losses / batch_idx)








def test_SNR_2_lpips(bidx, method, save_method):
    #이게 찐
    from QAM import Ps_qam, snr
    from SNR_VAE import add_noise_qam, add_noise_gau
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)

    lpips_losses = np.zeros((len(snr) + 1, len(np.arange(2, 28, 2)) + 1))
    with torch.no_grad():

        for batch_idx, (x_, y) in enumerate(validation_loader):
            print(f'Batch idx: [{batch_idx}/100]')
            x = x_.view(-1, x_dim)
            x = x.to(device)
            y = y.to(device)

            mu, logvar = model.encode(x)

            # TODO noise add to mu, logvar + Ax
            for avg in range(10):
                lpips_losses_avg = np.zeros((len(snr) + 1, len(np.arange(2, 28, 2)) + 1))
                for ber_index in np.arange(1,len(snr)+1):
                    # mu, logvar = add_noise_gau(mu, ber_index - 1), add_noise_gau(logvar, ber_index - 1)

                    z = model.reparameterize(mu, logvar)
                    z = add_noise_gau(z, ber_index-1)
                    recon_x = model.decode(z)

                    lpips_loss = LPIPS_MNIST(lpips_model, x.view(-1, 1, 28, 28), recon_x.view(-1, 1, 28, 28)).item()
                    lpips_losses_avg[ber_index-1, 0] += lpips_loss/batch_size

                    #TODO plot under-expessed term
                    for d in np.arange(2, 28, 2):
                        if 'SecondOrder' in save_method:
                            if type(bidx) == str:
                                A = torch.zeros((d, training_data.data.shape[1])).to(device)
                                for c in range(int(bidx)):
                                    A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                                A /= c
                            else:
                                A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)
                        else: #instan
                            A = calcul_A_MNIST(x, recon_x, d, True).to(device)

                        Ax = torch.matmul(A, x.view(-1, 28, 28))
                        Ax = add_noise_gau(Ax, ber_index - 1)
                        if 'SecondOrder' not in save_method: #instan
                            A = add_noise_gau(A, ber_index - 1)
                        A_ = torch.linalg.pinv(A)

                        final_recon_x = torch.matmul(A_, Ax) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                        lpips_loss = LPIPS_MNIST(lpips_model, x.view(-1, 1, 28, 28), final_recon_x.view(-1, 1, 28, 28)).item()
                        lpips_losses_avg[ber_index - 1, int(d/2)] += lpips_loss / batch_size
                lpips_losses += lpips_losses_avg/10

        if 'SecondOrder' in save_method:
            np.save(f'results/final_lpips_loss_{save_method}_bidx_{bidx}.npy', lpips_losses/batch_idx)
        else: #instan
            np.save(f'results/final_lpips_loss_{save_method}.npy', lpips_losses / batch_idx)










def test_BER(bidx, method, Pb):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses=np.zeros((13))
        for batch_idx, (x_, y) in enumerate(training_loader):

            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar, z, recon_x = model(x)
            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
            origin_mse_loss+= (reconst_loss/batch_size)

            #TODO plot under-expessed term
            d_loss = []
            for d in np.arange(14, 22, 2):
                if type(bidx)==str:
                    A = torch.zeros((d,training_data.data.shape[1])).to(device)
                    for c in range(int(bidx)):
                        A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                    A /= c
                else:
                    A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))


                Ax = Ax_error(Pb, Ax)
                A_ = torch.linalg.pinv(A)
                final_recon_x_receiv = torch.matmul(A_, Ax) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x_receiv, x.view(-1,28,28)).item()
                d_loss.append(final_reconst_loss/batch_size)
            d_losses += d_loss

        np.save('results/origin_mse_loss.npy', [origin_mse_loss/batch_idx])
        if type(bidx) == str:
            np.save(f'results/final_mse_loss_{method}_BER{Pb}_bidx_avg'+bidx+'.npy', d_losses/batch_idx)
        else:
            np.save(f'results/final_mse_loss_{method}_BER{Pb}_bidx_{bidx}.npy', d_losses / batch_idx)
        print('original loss: ', origin_mse_loss/batch_idx, '\n')
        print('final losses wrt d: ', d_losses/batch_idx)
        plt.plot(d_losses/batch_idx, label=f'Error prob: {Pb}')
        # plt.show()







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
                x = x.view(-1, x_dim)
                x = x.to(device)

                mu, logvar, z, recon_x = model(x)
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







def test_oneimg(bidx, method, save_method):
    with torch.no_grad():
        origin_mse_loss, final_mse_loss = 0.0, 0.0
        d_losses=np.zeros((14))
        ds=[]
        mse_loss=[]
        for batch_idx, (x_, y) in enumerate(validation_loader):
            print(f'batch: [{batch_idx}/{int(1000/batch_size)}]')
            x = x_.view(batch_size, x_dim)

            x = x.view(-1, x_dim)
            x = x.to(device)

            mu, logvar, z, recon_x = model(x)
            reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
            ds.append(0)
            mse_loss.append(reconst_loss/batch_size)


            #TODO plot under-expessed term
            d_loss = []
            d_loss.append(reconst_loss/batch_size)
            for d in np.arange(2, 28, 2):
                if 'SecondOrder' in save_method:
                    if type(bidx)==str:
                        A = torch.zeros((d,training_data.data.shape[1])).to(device)
                        for c in range(int(bidx)):
                            A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                        A /= c
                    else:
                        A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)
                else:
                    A = calcul_A_MNIST(x, recon_x, d,  True).to(device)

                Ax = torch.matmul(A, x.view(-1,28,28))
                A_ = torch.linalg.pinv(A)
                if 'Ax' in save_method:
                    final_recon_x = torch.matmul(A_, torch.matmul(A, x.view(-1, 28, 28))) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))
                else:
                    final_recon_x = torch.matmul(torch.matmul(A_, A), x.view(-1, 28, 28)) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                final_reconst_loss = nn.MSELoss()(final_recon_x, x.view(-1,28,28)).item()
                mse_loss.append(final_reconst_loss/batch_size)
                ds.append(d)
            d_losses += d_loss

            if batch_size*batch_idx > 1000:
                break

        if 'SecondOrder' in save_method:
            if type(bidx) == str:
                np.save(f'results/final_mse_loss_{save_method}_bidx_avg' + bidx + '.npy', d_losses / batch_idx)
            else:
                np.save(f'results/final_mse_loss_{save_method}_bidx_{bidx}.npy', d_losses / batch_idx)
        else:
            np.save(f'results/oneimg_final_mse_loss_{save_method}.npy', mse_loss)
            np.save(f'results/oneimg_ds_{save_method}.npy', ds)

        print('original loss: ', origin_mse_loss/batch_idx, '\n')
        print('final losses wrt d: ', d_losses/batch_idx)
        plt.plot(d_losses/batch_idx)
        plt.xticks(np.arange(0, 14, 1), np.arange(0, 28, 2))
        plt.xlabel('d')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error wrt d')







def test_oneimg_BER(bidx, method, batchidx, imgidx, Pbs, label):
    with torch.no_grad():
        for batch_idx, (x_, y) in enumerate(validation_loader):
            if batch_idx==batchidx:
                x = x_[imgidx].unsqueeze(0).view(1, x_dim)

                x = x.view(-1, x_dim)
                x = x.to(device)

                mu, logvar, z, recon_x = model(x)
                reconst_loss = nn.MSELoss()(recon_x.view(-1,28,28), x.view(-1,28,28)).item()
                origin_mse_loss = reconst_loss

                #TODO plot under-expessed term

                for d in np.arange(14, 22, 2):
                    if type(bidx)==str:
                        A = torch.zeros((d,training_data.data.shape[1])).to(device)
                        for c in range(int(bidx)):
                            A += torch.load(f'As/{method}/A_d{d}_bidx{c}.pt').to(device)
                        A /= c
                    else:
                        A = torch.load(f'As/{method}/A_d{d}_bidx{bidx}.pt').to(device)

                    d_loss = []
                    for Pb in Pbs:
                        Ax = torch.matmul(A, x.view(-1,28,28))
                        Ax = Ax_error(Pb, Ax)
                        A_ = torch.linalg.pinv(A)
                        final_recon_x_receiv = torch.matmul(A_, Ax) + torch.matmul((torch.eye(28).to(device) - torch.matmul(A_, A)), recon_x.view(-1, 28, 28))

                        final_reconst_loss = nn.MSELoss()(final_recon_x_receiv, x.view(-1,28,28)).item()
                        d_loss.append(final_reconst_loss)


                    plt.plot(d_loss, label=label + f'd= {d}')
        plt.xticks(np.arange(0, len(Pbs), 1), Pbs)
        plt.xlabel('Bit Error Probability')
        plt.ylabel('MSE loss')
        plt.title('Reconstruction error with BER wrt d')
        # plt.show()






if __name__ == "__main__":
    # Load data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.MNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
    validation_data = datasets.MNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
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
    hidden_dim = 128 # 128  #16(32) 1040(64) 520(128) 256(batch1, 256) 324(512) 296(1040)
    model_method = f'VAE'
    method= f'VAE'
    save_method= f'VAE'
    bidx = 100 #'10'


    # Load model
    model = VAE(x_dim, hidden_dim).to(device)
    try:
        model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/'+model_method+'.pth'))
    except:
        print('wrong model')
        model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/' + 'VAE' + '.pth'))

    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))

    test(bidx=bidx, method=method, save_method=save_method)
    # test_semantic(bidx=bidx, method=method, save_method=save_method, classifier=classifier)
    # test_SNR_2(bidx=bidx, method=method, save_method=save_method) #snr넣은 instan 모델, method로 하고, second할때 savemethod만 SecondOrder 있게 바꾸면됨
    # test_SNR_2_lpips(bidx=bidx, method=method, save_method=save_method)

    # test_SNR_A(bidx=bidx, method=method, save_method=save_method)
    # test_oneimg(bidx, method, save_method)





    # plt.legend()
    # plt.show()



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
