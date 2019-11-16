import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# Used to initialize the weights of yhe Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = torch.nn.Linear(4096, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 7)
        
        self.dropout50 = nn.Dropout(0.50)
        
        I.xavier_normal_(self.conv1.weight, gain=1)
        I.xavier_normal_(self.conv2.weight, gain=1)
        I.xavier_normal_(self.conv3.weight, gain=1)
        I.xavier_normal_(self.conv4.weight, gain=1)
        I.xavier_normal_(self.fc1.weight, gain=1)
        I.xavier_normal_(self.fc2.weight, gain=1)
        I.xavier_normal_(self.fc3.weight, gain=1)

        
    def forward(self, x):
        # Size changes from (3, 32, 32) to (32, 32, 32)
        #print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        #print(x.shape)

        # Size changes from (32, 32, 32) to (32, 32, 32)
        x = F.leaky_relu(self.conv2(x))
        #print(x.shape)

        # Size changes from (32, 32, 32) to (32, 16, 16)
        x = self.pool(x)
        #print(x.shape)

        # Size changes from (32, 16, 16) to (64, 16, 16)
        x = F.leaky_relu(self.conv3(x))
        #print(x.shape)
        
        # Size changes from (64, 16, 16) to (64, 16, 16)
        x = F.leaky_relu(self.conv4(x))
        #print(x.shape)

        # Size changes from (64, 16, 16) to (64, 8, 8)    
        x = self.pool(x)
        #print(x.shape)

        # Size changes from (64, 8, 8) to (4096)
        x = x.view(x.size(0), -1)
        
        # Size changes from (4096) to (512)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.dropout50(x)
        
        # Size changes from (512) to (512)
        x = F.relu(self.fc2(x))
        x = self.dropout50(x)
        
        # Size changes from (512) to (6)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
