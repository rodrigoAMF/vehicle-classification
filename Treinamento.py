import torch
print(torch.__version__)

# Base libs
import os
import torch
import numpy as np

# Preprocess dataset
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Show images and graphs
import matplotlib.pyplot as plt

# Build ANN
from torch import nn
from torch import optim
import torch.nn.functional as F

# Open images from folder
from PIL import Image

data_dir = "dataset/"

# Número de subprocessos para se usar ao carregar os dados (0 nenhum só programa principal)
num_workers = 0
# Quantidade de imagens por 'batch' ao carregar
batch_size = 20
# Porcentagem do conjunto de treinamento usado para validação
valid_size = 0.25

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

data = datasets.ImageFolder(data_dir, transform=transform)

# Obtain training indices that will be used for validation
num_dados = len(data)
indices = list(range(num_dados))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_dados))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)

# Mapeamento das classes
classes = {}
for idx, value in data.class_to_idx.items():
    classes[value] = idx

# Construindo um classificador

