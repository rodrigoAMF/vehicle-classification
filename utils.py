import os
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
from model import Net
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
         # linear layer (1024 -> 512)
        self.fc1 = nn.Linear(2048, 1024)
        # linear layer (512 -> 256)
        self.fc2 = nn.Linear(1024, 256)
        # linear layer (256 -> 102)
        self.fc3 = nn.Linear(256, 7)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

## Functions used on split dataset ##
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

## Functions used on model training ##

# helper function to un-normalize an image
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# helper function to display an image
def imshow(img):
    img = torch.tensor(img)
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img = unorm(img)
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    
## Functions used to evaluate the model ##

def load_model(path, verbose=True):
    model = Net()
    if os.path.isfile(path):
        print("Loading model from %s ..." % path)
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        model.transform = checkpoint['transform']
        
        epoch         = checkpoint['epoch']
        learning_rate = checkpoint['learning_rate']
        batch_size    = checkpoint['batch_size']
        criterion     = checkpoint['criterion']
        train_loss    = checkpoint['train_loss']
        valid_loss    = checkpoint['valid_loss']
   
        train_acc     = checkpoint['train_acc']
        valid_acc     = checkpoint['valid_acc']
        train_prec    = checkpoint['train_prec']
        valid_prec    = checkpoint['valid_prec']
        train_recall  = checkpoint['train_recall']
        valid_recall  = checkpoint['valid_recall']
        
        labels_map    = checkpoint['labels_map']
        
        print("Model loaded!")
        if verbose:
            print(model)
            print("Model trained until epoch %s with learning rate %f and batch_size %d" % (epoch, learning_rate, batch_size))
            print('Training Loss: {:.6f}      \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
            print('Training Accuracy: {:.2f}%\tValidation Accuracy: {:.2f}%'.format(train_acc, valid_acc))
            print('Training Precision: {:.2f}%\tValidation Precision: {:.2f}%'.format(train_prec, valid_prec))
            print('Training Recall: {:.2f}%\t\tValidation Recall: {:.2f}%'.format(train_recall, valid_recall))
    else:
        print("No model found at %s" % path)

    return model, labels_map, criterion

def load_resnet50_modified(path, verbose=True):
    model = models.resnet50(pretrained=True)
    classifier = Classifier()
    model.fc = classifier
    if os.path.isfile(path):
        print("Loading model from %s ..." % path)
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        model.transform = checkpoint['transform']
        
        epoch         = checkpoint['epoch']
        learning_rate = checkpoint['learning_rate']
        batch_size    = checkpoint['batch_size']
        criterion     = checkpoint['criterion']
        train_loss    = checkpoint['train_loss']
        valid_loss    = checkpoint['valid_loss']
   
        train_acc     = checkpoint['train_acc']
        valid_acc     = checkpoint['valid_acc']
        train_prec    = checkpoint['train_prec']
        valid_prec    = checkpoint['valid_prec']
        train_recall  = checkpoint['train_recall']
        valid_recall  = checkpoint['valid_recall']
        
        labels_map    = checkpoint['labels_map']
        
        print("Model loaded!")
        if verbose:
            print(model)
            print("Model trained until epoch %s with learning rate %f and batch_size %d" % (epoch, learning_rate, batch_size))
            print('Training Loss: {:.6f}      \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
            print('Training Accuracy: {:.2f}%\tValidation Accuracy: {:.2f}%'.format(train_acc, valid_acc))
            print('Training Precision: {:.2f}%\tValidation Precision: {:.2f}%'.format(train_prec, valid_prec))
            print('Training Recall: {:.2f}%\t\tValidation Recall: {:.2f}%'.format(train_recall, valid_recall))
    else:
        print("No model found at %s" % path)

    return model, labels_map, criterion

def getModelPredicitons(model, loader, criterion=None):
    if criterion is not None:
        loss_total = 0
    y_true = []
    y_pred = []

    model.cpu()
    model.eval()
    # iterate over test data
    for data, target in loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        _, pred = torch.max(output, 1)
        if criterion is not None:
            loss = criterion(output, target)
            loss_total += loss.item()*data.size(0)
        y_true.append(target.data.cpu())
        y_pred.append(pred.data.cpu())
    y_true = np.array(torch.cat(y_true))
    y_pred = np.array(torch.cat(y_pred))
    
    if criterion is not None:
        return y_true, y_pred, loss
    return y_true, y_pred

def getAccuracyPerClass(model, loader, num_classes):
    class_correct = list(0 for i in range(num_classes))
    class_total = list(0 for i in range(num_classes))

    model.eval()

    # iterate over test data
    for data, target in loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        # calculate test accuracy for each object class
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    class_correct = np.array(class_correct)
    class_total = np.array(class_total)
    accuracy = (class_correct/class_total)*100.0
    
    return accuracy, class_correct, class_total

def printAccuracyPerClass(class_correct, class_total, labels_map, split_name="Validation"):
    for i in range(len(class_total)):
        percentage_correct = (class_correct[i]/class_total[i])*100.0
        print("%s accuracy of %-12s %.2f%% (%d/%d)" % (split_name, labels_map[i], percentage_correct, class_correct[i], class_total[i]))
    #percentage_total = (class_correct.sum()/class_total.sum()) * 100.0
    #print("\n%s accuracy (Overall): %.2f%% (%d/%d)" % (split_name, percentage_total, class_correct.sum(), class_total.sum()))
    
def plotConfusionMatrix(confusion_matrix, labels_map):
    class_names = list(labels_map.values())
    df_cm = pd.DataFrame(confusion_matrix, index = class_names,
                      columns = class_names)
    df_corr = df_cm.corr()
    plt.figure(figsize = (15,11))
    ax = sns.heatmap(df_cm, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.yticks(va="center")
    plt.show()
    
def getMetricsDataframe(accuracy, precision, recall, f1_score, labels_map):
    metrics = np.empty([4, len(labels_map)+1])
    for i in range(metrics.shape[1]):
        metrics[0][i] = accuracy[i]
    for i in range(metrics.shape[1]):
        metrics[1][i] = precision[i]
    for i in range(metrics.shape[1]):
        metrics[2][i] = recall[i]
    for i in range(metrics.shape[1]):
        metrics[3][i] = f1_score[i]

    index_names = list(labels_map.values())
    index_names.append("Average")
    metrics = pd.DataFrame(metrics.transpose(), columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=index_names)
    metrics = metrics.applymap("{0:.2f}%".format)
    
    return metrics

def process_image(image_path):
    mean = np.array([0.485, 0.456, 0.406]) # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    
    img = Image.open(image_path)
    # Checks whether height or width is larger, then resizes (10000 can be big number)
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    
    # Crop getting the differences between 244-each corner
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    img = np.array(img)/255
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def getProbClasses(output, classes):
    probabilities = np.squeeze(np.array(output))

    probabilities_dict = {}
    for i in range(len(probabilities)):
        probabilities_dict[probabilities[i]] = classes[i]

    probabilities = np.sort(probabilities)

    top_classes = []
    for prob in probabilities:
        top_classes.append(probabilities_dict[prob])
        
    return probabilities, top_classes

def getImagePredictions(image_path, model, labels_map):
    classes = list(labels_map.values())
    
    image = torch.tensor(process_image(image_path))
    image = image.view(1, 3, 224, 224).float()

    model.eval()
    with torch.no_grad():
        output = model(image)
    sm = torch.nn.Softmax(dim=1)
    output = sm(output).cpu()

    probabilities, top_classes = getProbClasses(output, classes)
    
    fig = plt.figure(figsize = (6,10))

    plt.subplot(2, 1, 1)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("car")

    plt.subplot(2, 1, 2)
    y_pos = np.arange(len(probabilities))
    plt.barh(y_pos, probabilities)
    plt.yticks(y_pos, top_classes)

    plt.show()
    
def plot_filters_single_channel_big(t):
    
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]
    
    
    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    
    npimg = npimg.T
    
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)
    
def plot_filters_single_channel(t):
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()
    
def plot_filters_multi_channel(t):
    
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()
