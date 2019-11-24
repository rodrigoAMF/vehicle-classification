import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

def load_data(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data = datasets.ImageFolder(data_dir, transform=transform)
    classes = data.classes
    indices = list(range(len(data)))

    sampler = SubsetRandomSampler(indices)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, num_workers=0)
    
    return loader, classes

def show_numpy_image(image):
    # Expect image with shape transposed(1,2,0)
    plt.imshow(image)
    plt.show()
    
def save_image(path, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)*255.0
    cv2.imwrite(path + ".png", rgb_image)
    
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)