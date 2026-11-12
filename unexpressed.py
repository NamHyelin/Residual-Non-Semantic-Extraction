import torch
import sys
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from VAE_model import VAE
from classifier import CNN
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calcul_A_CIFAR(data, data_recon, d, large):
    # d < data.size(1)
    sub = data - data_recon
    sub_reshaped = sub.view(-1, 28, 28)
    sub_T = (data - data_recon).permute(0, 1, 3, 2)

    sub_reshaped = sub.view(-1, 28, 28)
    sub_T_reshaped = sub_T.view(-1, 28, 28)
    R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
    R = R_reshaped.view(data.size())
    R = torch.mean(R, 0)

    eigenvalues, eigenvectors = torch.linalg.eig(R.float())  # (3,32), #(3,32,32)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    _, indices = torch.topk(eigenvalues, k=d, dim=1, largest=large)  # (3,5)

    D = data.size()[0]
    c, s = eigenvalues.size()
    A = torch.zeros((c, s, d))
    for j in range(c):
        for k in range(d):
            index = indices[j, k]
            A[j, :, k] = eigenvectors[j, :, index]
    A = A.repeat(D, 1, 1, 1)
    A = A.permute(0, 1, 3, 2)
    return A


def calcul_A_eigenvalues(data, data_recon):
    sub = data - data_recon
    sub_reshaped = sub.view(-1, 28, 28)
    sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

    R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
    R_reshaped = torch.mean(R_reshaped, 0)  # average loss

    eigenvalues, eigenvectors = torch.linalg.eig(R_reshaped.float())  # (28), #(28,28)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    eigenvalues[eigenvalues.abs() < 1e-5] = 0
    num_nonzero_ev = torch.count_nonzero(eigenvalues)
    return num_nonzero_ev



def calcul_A_MNIST(data, data_recon, d, large):
    # d < data.size(1)
    sub = data - data_recon
    sub_reshaped = sub.view(-1, 28, 28)
    sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

    R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
    R_reshaped = torch.mean(R_reshaped, 0)  # average loss

    eigenvalues, eigenvectors = torch.linalg.eig(R_reshaped.float())  # (28), #(28,28)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    # eigenvalues[eigenvalues.abs() < 1e-5] = 0

    _, indices = torch.topk(eigenvalues, k=d, dim=0, largest=large)
    D = data.size()[0]
    s = eigenvalues.size()[0]
    A = torch.zeros((s, d))
    for k in range(d):
        index = indices[k]
        A[:, k] = eigenvectors[:, index]
    # A = A.repeat(D, 1, 1)
    # A = A.permute(0, 2, 1)
    A = A.transpose(0, 1)
    return A




def calcul_A_MNIST_imaginary(data, data_recon, d, large):
    # d < data.size(1)
    sub = data - data_recon
    sub_reshaped = sub.view(-1, 28, 28)
    sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

    R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
    R_reshaped = torch.mean(R_reshaped, 0)  # average loss

    eigenvalues, eigenvectors = torch.linalg.eig(R_reshaped.float())  # (28), #(28,28)
    eigenvalues_r, eigenvectors_r = eigenvalues.real, eigenvectors.real

    _, indices = torch.topk(eigenvalues_r, k=d, dim=0, largest=large)
    D = data.size()[0]
    s = eigenvalues.size()[0]
    A = torch.zeros((s, d))
    for k in range(d):
        index = indices[k]
        A[:, k] = eigenvectors[:, index]
    # A = A.repeat(D, 1, 1)
    # A = A.permute(0, 2, 1)
    A = A.transpose(0, 1)
    return A






def calcul_A_singularvalues(data, data_recon ,m):
    sub = data - data_recon
    sub_reshaped = sub.view(-1, m, int(784/m))
    sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

    R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
    R_reshaped = torch.mean(R_reshaped, 0)  # average loss

    singularvectors, singularvalues, _ = torch.linalg.svd(R_reshaped.float())  # (28), #(28,28)
    singularvalues, singularvectors = singularvalues.real, singularvectors.real
    singularvalues = torch.pow(singularvalues, 2)

    singularvalues[singularvalues.abs() < 1e-5] = 0
    num_nonzero_sv = torch.count_nonzero(singularvalues)
    return num_nonzero_sv



def calcul_A_singular_MNIST(data, data_recon, m, d, large):
    # d < data.size(1)
    sub = data - data_recon
    sub_reshaped = sub.view(-1, m, int(784/m))
    sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

    R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
    R_reshaped = torch.mean(R_reshaped, 0)  # average loss

    singularvectors, singularvalues, _ = torch.linalg.svd(R_reshaped.float())  # (28), #(28,28)
    singularvalues, eigenvectors = singularvalues.real, singularvectors.real
    # eigenvalues[eigenvalues.abs() < 1e-5] = 0

    _, indices = torch.topk(singularvalues, k=d, dim=0, largest=large)
    D = data.size()[0]
    s = singularvalues.size()[0]
    A = torch.zeros((s, d))
    for k in range(d):
        index = indices[k]
        A[:, k] = singularvectors[:, index]
    # A = A.repeat(D, 1, 1)
    # A = A.permute(0, 2, 1)
    A = A.transpose(0, 1)
    return A





def calcul_A_ImageNet(data, data_recon, d, large):
    # d < data.size(1)
    data = torch.mean(data, 1)
    data_recon = torch.mean(data_recon, 1)
    sub = data - data_recon
    sub_reshaped = sub.view(-1, 224, 224)
    sub_T_reshaped = sub_reshaped.permute(0, 2, 1)

    R_reshaped = torch.bmm(sub_reshaped, sub_T_reshaped)
    R_reshaped = torch.mean(R_reshaped, 0)  # average loss

    eigenvalues, eigenvectors = torch.linalg.eig(R_reshaped.float())  # (28), #(28,28)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    # eigenvalues[eigenvalues.abs() < 1e-5] = 0

    _, indices = torch.topk(eigenvalues, k=d, dim=0, largest=large)
    D = data.size()[0]
    s = eigenvalues.size()[0]
    A = torch.zeros((s, d))
    for k in range(d):
        index = indices[k]
        A[:, k] = eigenvectors[:, index]
    # A = A.repeat(D, 1, 1)
    # A = A.permute(0, 2, 1)
    A = A.transpose(0, 1)
    return A





def transmit_A(A, test_data):
    # A = calcul_A(test_data, test_data_recon, d, True)
    A_ = torch.linalg.pinv(A)
    plot_transmit = torch.matmul(torch.matmul(A_, A).to(device),test_data)
    add_transmit = torch.matmul(A, test_data)
    final_data = torch.matmul(torch.matmul(A_,add_transmit)) + torch.matmul((torch.eye(A_.size(0))-torch.matmul(A_,A)), test_data) # 인지 torch.matmul(torch.matmul(A_, A).to(device),test_data)
    # unexp_test_recon_error = F.mse_loss((test_data_recon+add_transmit)/2, test_data) / test_data_variance
    return A, add_transmit, final_data





if __name__ == "__main__":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    # datasets.CIFAR10.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    training_data = datasets.FashionMNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                          transform=transforms.Compose([transforms.ToTensor()]))  # , transforms.Normalize((0.1307), (0.3081))
    validation_data = datasets.FashionMNIST(root="C:/TransUnrecon/data", train=False, download=True,
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

    # Parameters
    x_dim = training_data.data.shape[1] ** 2
    hidden_dim = 128
    learning_rate = 1e-3
    num_epochs = 0
    num_warmup_epochs = 50

    method = 'VAE_diffN'
    model_method = 'VAE'

    # Load model
    model = VAE(x_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/'+model_method+'.pth'))



    for batch_idx_tr, (x, _) in enumerate(training_loader):
        model.train()
        x = x.view(-1, x_dim)
        x = x.to(device)

        mu, logvar, z, recon_x = model(x)


        # TODO calculate A
        for d in np.arange(2, 28, 2):
            x, recon_x = x.cpu(), recon_x.cpu()
            A = calcul_A_MNIST(x, recon_x, d, True).cpu()  # (d, 28)
            try:
                torch.save(A, f'As/{method}/A_d{d}_bidx{batch_size}.pt')
            except:
                os.makedirs(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}')
                torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{batch_size}.pt')


        if batch_idx_tr == 0:
            sys.exit(0)