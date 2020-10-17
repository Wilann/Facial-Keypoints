## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        '''
        ATTEMPT #1:
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1)  
        self.fc1 = nn.Linear(29*29*128, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        x = F.relu(self.conv1(x))            # (batch_size, 224, 224, 1)  ->  (batch_size, 113, 113, 32)
        x = F.relu(self.conv2(x))            # (batch_size, 113, 113, 32) ->  (batch_size, 57, 57, 64) 
        x = F.relu(self.conv3(x))            # (batch_size, 57, 57, 64)   ->  (batch_size, 29, 29, 128)
        x = x.reshape(-1, 29*29*128)         # (batch_size, 29, 29, 128)  ->  (batch_size, 29*29*128)
        x = self.fc1(x)
        x = self.fc2(x)                      # (batch_size, 29*29*128)    ->  (batch_size, 1000)
        x = F.dropout(self.fc3(x), p=0.4     # (batch_size, 1000) -> (batch_size, 136)
        '''
        
        self.conv1 = nn.Conv2d(1, 68, 3)
        self.conv2 = nn.Conv2d(68, 136, 3)
        self.conv3 = nn.Conv2d(136, 272, 3)
        self.conv4 = nn.Conv2d(272, 544, 3)
        self.conv5 = nn.Conv2d(544, 1088, 3)
        self.conv6 = nn.Conv2d(1088, 2176, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2176*1*1, 1000) 
        self.fc2 = nn.Linear(1000, 1000)  
        self.fc3 = nn.Linear(1000, 136)
        self.dropout = nn.Dropout(p=0.4)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    