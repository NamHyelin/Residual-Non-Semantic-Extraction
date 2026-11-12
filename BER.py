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

from classifier import CNN
from train_VAE import VAE, get_gauss, train_loss, Tsne_idx
from unexpressed import calcul_A_MNIST



def Ax_error(Pb, Ax):
    max_value = torch.max(Ax)
    min_value = torch.min(Ax)

    e=0
    for i in range(Ax.size(1)):
        for j in range(Ax.size(2)):
            if torch.rand(1).item() < Pb:
                Ax[:, i, j] = torch.FloatTensor(1).uniform_(min_value, max_value/2).item()
                e+=1
    print(f'Error bits out of {Ax.size(1)}x{Ax.size(2)} bits: {e}')

    return Ax