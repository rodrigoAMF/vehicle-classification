import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# Used to initialize the weights of yhe Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0.01)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0.01)
        
        '''self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0.01)'''
        
        '''self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0.01)'''
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = torch.nn.Linear(16384, 2098)
        self.fc2 = torch.nn.Linear(2098, 512)
        self.fc3 = torch.nn.Linear(512, 7)
        
        self.dropout50 = nn.Dropout(0.50)
        
        #I.xavier_normal_(self.conv1.weight, gain=1)
        #I.xavier_normal_(self.conv2.weight, gain=1)
        #I.xavier_normal_(self.conv3.weight, gain=1)
        #I.xavier_normal_(self.conv4.weight, gain=1)
        #I.xavier_normal_(self.fc1.weight, gain=1)
        #I.xavier_normal_(self.fc2.weight, gain=1)
        #I.xavier_normal_(self.fc3.weight, gain=1)

        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        
        x = self.pool(x)

        x = F.leaky_relu(self.conv2(x))

        x = self.pool(x)

        '''x = F.leaky_relu(self.conv3(x))
        
        x = self.pool(x)
        
        x = F.leaky_relu(self.conv4(x))

        x = self.pool(x)'''

        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout50(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout50(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
