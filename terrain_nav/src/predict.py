#!/usr/bin/env python

import os
import torch
from torch import FloatTensor
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
    
class TCNet32(nn.Module):
    def __init__(self):
        super(TCNet32, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.fc1 = nn.Linear(128 * 1 * 1, 128)
        # self.fc_action = nn.Linear(1, 16)
        # self.fc2 = nn.Linear(128 + 16, 64)
        # self.fc3 = nn.Linear(64, 1)
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 1 * 1, 128),
            nn.Dropout(0.5)
        )
        self.fc_action = nn.Linear(1, 16)
        self.fc2 = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, action):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        action = F.relu(self.fc_action(action.view(-1, 1)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class TCNetRes(nn.Module):
    def __init__(self):
        super(TCNetRes, self).__init__()
        self.layer1 = self._make_layer(1, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.layer4 = self._make_layer(64, 128, stride=2)
        
        self.fc1 = nn.Linear(512, 128)
        self.fc_action = nn.Linear(1, 16)
        self.fc2 = nn.Linear(128 + 16, 64)
        self.fc3 = nn.Linear(64, 1)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        return nn.Sequential(*layers)
        
    def forward(self, x, action):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        action = F.relu(self.fc_action(action.view(-1, 1)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = TCNetRes()
net.load_state_dict(torch.load('res_rot_dist_20230722_165815_7570'))
net.to(device)
net.eval()

input_image_path = '9.png'
input_image = Image.open(input_image_path).convert('L')
input_image = Image.fromarray(np.array(input_image))
transform = transforms.Compose([
        transforms.ToTensor()
])

image = transform(input_image).unsqueeze(0).to(device)
action_dict = {'left': 0, 'middle': 1, 'right': 2}

for action in action_dict:
    action_value = torch.tensor([action_dict[action]], dtype=torch.float).to(device)

    with torch.no_grad():
        prediction = net(image, action_value)
        
    mean =  4.089976655556117 #16623.02530253025
    std =  7.511511417643042 #4212.291661087521
    
    #prediction = (prediction * std) + mean
    
    print(f'{action}: {prediction.item()}')


