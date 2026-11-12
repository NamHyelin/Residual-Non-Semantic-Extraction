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
from unexpressed import calcul_A_MNIST
from lpips_score import LPIPS_MNIST
from QAM import Ps_qam, snr
from SNR_VAE import add_noise_qam, add_noise_gau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class VAE(torch.nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()


        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(2048, latent_dim)  # latent_dim *2 < inputdim x inputdim
        self.encoder_logvar = torch.nn.Linear(2048, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim),
            torch.nn.Sigmoid(),
        )


    def encode(self, x):
        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return mu, logvar, z, recon_x



def train_loss(batch_size, x, recon_x, mu_1, logvar_1, mu_2 = None, logvar_2 = None):
    if mu_2 == None and logvar_2==None:
        mu_2 = torch.zeros_like(mu_1)
        logvar_2 = torch.log(torch.ones_like(logvar_1))
    reconst_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size # nn.MSELoss()(recon_x, x) # nn.BCEWithLogitsLoss()(recon_x, x) / batch_size
    regular_loss = torch.mean(0.5 * ((1/(logvar_2.exp()))*((mu_1-mu_2).pow(2)) + (logvar_1.exp())/(logvar_2.exp())  - 1 - (logvar_1-logvar_2))) #sum 막해도되는지
    return reconst_loss, regular_loss




def Tsne_idx(latents):
    num_samples, latent_dim = latents.size()
    tsne = TSNE(n_components=2, n_iter=500, perplexity=2)  #n_iter=1000 #30 if num_samples>30 else 10
    latent_np = latents.clone().detach().cpu().numpy()
    digits_tsne = tsne.fit_transform(latent_np)

    eps = (digits_tsne[:, 0].max() - digits_tsne[:, 0].min()) / 15 #30
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(digits_tsne)
    dbscan_labels -= dbscan_labels.min()

    num_clusters = dbscan_labels.max()+1 if dbscan_labels.max() > 0 else 1
    cluster_center_latents = np.zeros((num_clusters, latent_dim))
    for i in range(num_clusters):
        indices = np.where(dbscan_labels == i)[0]
        cluster_center_latents[i] = torch.mean(latents[indices]).item()
    return dbscan_labels, cluster_center_latents, num_clusters





def get_gauss(dbscan_labels, num_clusters, mu, logvar):
    num_samples, latent_dim = mu.size()

    mu_s = torch.zeros((num_clusters, latent_dim))
    logvar_s = torch.zeros((num_clusters, latent_dim))
    for i in range(num_clusters):
        indices = np.where(dbscan_labels == i)[0]
        mu_s[i] = torch.mean(mu[indices])
        logvar_s[i] = torch.mean(logvar[indices])

    mu_2 = torch.zeros_like(mu)
    logvar_2 = torch.zeros_like(logvar)
    for batch in range(num_samples):
        mu_2[batch] = mu_s[dbscan_labels[batch]]
        logvar_2[batch] = logvar_s[dbscan_labels[batch]]

    return mu_2, logvar_2



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
    num_epochs = 10 #0
    num_warmup_epochs = 0  #50
    x_dim = training_data.data.shape[1]**2
    hidden_dim = 128 #하고싶은거의 반값을 넣으면됨
    learning_rate = 1e-3
    method= f'VAE_denoising2'



    # Load model
    model = VAE(x_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(f'C:/TransUnrecon/VAE_MNIST_2/{method}.pth'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    reconst_loss_train, regular_loss_train, Bs = [], [], []
    reconst_loss_test, regular_loss_test, semantic_loss_test, num_clusters_test = [], [], [], []
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load('C:/TransUnrecon/VAE_MNIST_2/classifier_mnist.pth'))
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)

    for e in range(num_epochs + num_warmup_epochs):
        reconst_loss_tr, regular_loss_tr = 0.0, 0.0
        reconst_loss_ts, regular_loss_ts, semantic_loss_ts, num_clusters_ts = 0.0, 0.0, 0.0, 0.0

        for batch_idx_tr, (x, _) in enumerate(training_loader):
            model.train()
            x = x.view(-1, x_dim)
            x = x.to(device)

            optimizer.zero_grad()
            if 'denoising' in method:
                ber_index = np.random.randint(1, len(snr) + 1)
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar)
                z = add_noise_gau(z, ber_index - 1)
                recon_x = model.decode(z)
            else:
                mu, logvar, z, recon_x = model(x)


            if e < num_warmup_epochs:
                B = 0.5 #0.5
                reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x, recon_x=recon_x, mu_1=mu, logvar_1=logvar)
                loss = (1 - B) * reconst_loss + B * regular_loss
            else:
                # B = 0.1 * e
                # dbscan_labels, cluster_center_latents, num_clusters_tr = Tsne_idx(z)
                # mu_2, logvar_2 = get_gauss(dbscan_labels, num_clusters_tr, mu, logvar)
                # reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x, recon_x=recon_x, mu_1=mu, logvar_1=logvar, mu_2=mu_2, logvar_2=logvar_2)
                # loss = (1 - B) * reconst_loss + B * regular_loss
                B=0
                reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x, recon_x=recon_x, mu_1=mu, logvar_1=logvar)
                loss = (1 - B) * reconst_loss + B * regular_loss


            reconst_loss_tr += reconst_loss.item()
            regular_loss_tr += regular_loss.item()

            loss.backward()
            optimizer.step()


            #TODO calculate A
            if e == num_epochs + num_warmup_epochs-1 :
                print('saving A for batch ', batch_idx_tr)
                for d in np.arange(2, 28, 2):
                    x, recon_x = x.cpu(), recon_x.cpu()
                    if '_A' in method:
                        # smalleigen
                        A = calcul_A_MNIST(x, recon_x, d, False).cpu()
                    else:
                        A = calcul_A_MNIST(x, recon_x, d, True).cpu()  # (d, 28)
                    try:
                        torch.save(A, f'As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')
                    except:
                        os.makedirs(f'C:/TransUnrecon/VAE_MNIST_2/As/{method}')
                        torch.save(A, f'C:/TransUnrecon/VAE_MNIST_2/As/{method}/A_d{d}_bidx{batch_idx_tr}.pt')





        with torch.no_grad():
            for batch_idx_ts, (x, label) in enumerate(validation_loader):
                x_ = x.view(batch_size, x_dim)
                x, x_, label = x.to(device), x_.to(device), label.to(device)
                mu, logvar, z, recon_x = model(x_)
                # dbscan_labels, cluster_center_latents, num_clusters = Tsne_idx(z)
                # mu_2, logvar_2 = get_gauss(dbscan_labels, num_clusters, mu, logvar)
                reconst_loss, regular_loss = train_loss(batch_size=batch_size, x=x_, recon_x=recon_x, mu_1=mu, logvar_1=logvar)

                # classifier_loss = nn.CrossEntropyLoss()(classifier(recon_x.view(batch_size, 1, 28, 28)), label)
                lpips_score = LPIPS_MNIST(lpips_model, x.view(-1,1,28,28), recon_x.view(-1,1,28,28))

                reconst_loss_ts += reconst_loss.item()
                regular_loss_ts += regular_loss.item()
                semantic_loss_ts += lpips_score.item() #classifier_loss.item()
                # num_clusters_ts += num_clusters

        reconst_loss_train.append(reconst_loss_tr / batch_idx_tr)
        # regular_loss_train.append(regular_loss_tr / batch_idx_tr)
        reconst_loss_test.append(reconst_loss_ts / batch_idx_ts)
        regular_loss_test.append(regular_loss_ts / batch_idx_ts)
        semantic_loss_test.append(semantic_loss_ts / batch_idx_ts)
        # num_clusters_test.append(num_clusters_ts / batch_idx_ts)
        # Bs.append(B)

        print(f'[{e} / {num_epochs + num_warmup_epochs}] reconstruction loss: {reconst_loss_test[-1]}, regularization loss: {regular_loss_test[-1]}, semantic loss: {semantic_loss_test[-1]}')



        np.save('results/reconst_loss_train_'+ method + '.npy', reconst_loss_train)
        # np.save('results/regular_loss_train_' + method + '.npy', regular_loss_train)
        np.save('results/reconst_loss_test_' + method + '.npy', reconst_loss_test)
        # np.save('results/regular_loss_test_' + method + '.npy', regular_loss_test)
        np.save('results/semantic_loss_test_' + method + '.npy', semantic_loss_test)
        # np.save('results/num_clusters_test_' + method + '.npy', num_clusters_test)
        # np.save('results/B_' + method + '.npy', Bs)
        torch.save(model.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/'+method+'.pth')