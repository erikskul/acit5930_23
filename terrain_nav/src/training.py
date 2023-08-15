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

# Define your dataset class
class TerrainDataset(Dataset):
    # Define the constructor, annotations is non-spacial data, directory is the root directory of the images
    def __init__(self, annotations, directory, column_index):
        # Load the directories
        self.directory = directory
        self.column_index = column_index
        annotations_file = os.path.join(self.directory, annotations)
        # Load non-spatial data of the images
        self.nonspatial_data = pd.read_csv(annotations_file)
        # Load the transform
        self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(20),  # Rotation within +/- 20 degrees
                                        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),  # Optional random erasing                                   
                                        ])
        
        # self.transform = transforms.Compose([
        #                                 transforms.ToTensor(),
        #                                 transforms.Resize((20, 20)),  # Scale the image
        #                                 transforms.Pad(2),  # Pad the image
        #                                 transforms.RandomCrop((16, 16)),  # Random crop to simulate translation
        #                                 transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),  # Optional random erasing                                   
        #                                 ])
        
        # self.transform = transforms.Compose([
        #                                 transforms.ToTensor(),
        #                                 ])
        self.action_dict = {'left': 0, 'middle': 1, 'right': 2}

        # Calculate mean and std of the labels
        self.label_mean = self.nonspatial_data.iloc[:, self.column_index].mean()
        self.label_std = self.nonspatial_data.iloc[:, self.column_index].std()
        print(f"Label mean: {self.label_mean}, label std: {self.label_std}")

    def __len__(self):
        # Return the length of the dataset
        return len(self.nonspatial_data)
    
    def get_stats(self):
        return self.label_mean, self.label_std
        
    def __getitem__(self, idx):
        # Return a data sample and its corresponding label 
        # Define the path of the image
        img_path = os.path.join(self.directory, 'images', self.nonspatial_data.iloc[idx, 0])
        # Load the image
        image = Image.open(img_path).convert('L')
        # Load the action
        action_str = self.nonspatial_data.iloc[idx, 1]
        action = self.action_dict[action_str]
        # Load the label and normalize it
        label = self.nonspatial_data.iloc[idx, self.column_index]
        # label = (label - self.label_mean) / self.label_std
        image = Image.fromarray(np.array(image))
        image = self.transform(image)

        return image, action, label
      
class TCNet16(nn.Module):
    def __init__(self):
        super(TCNet16, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Change here: 128 * 1 * 1 instead of 128 * 2 * 2
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc_action = nn.Linear(1, 3)
        self.fc2 = nn.Linear(32 + 3, 16)
        self.fc3 = nn.Linear(16, 1)

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

# Define your dataset directories
all_data_dir = '/home/user/scout_ws/dataset2ndDistanceBlack'
annotations = 'annotations.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 100
learning_rate = 0.00001

# Set seed
torch.manual_seed(13)

# Initialize dataset
spawn_dataset = TerrainDataset(annotations, all_data_dir, 4) # 4 = energy / 2d distance, 3 = 2d distance, 2 = energy
full_dataset = DataLoader(spawn_dataset, batch_size=batch_size, shuffle=True)
train_size = int(0.8 * len(full_dataset.dataset))
val_size = len(full_dataset.dataset) - train_size
# Split dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset.dataset, [train_size, val_size])

# Create dataloader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Report split sizes
print('Training set has {} instances'.format(len(train_dataset)))
print('Validation set has {} instances'.format(len(val_dataset)))

# Create model object
net = TCNetRes().to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5) #SGD, momentum=0.9


def unnorm_labels(labels):
    mean, std = spawn_dataset.get_stats()
    return labels * std + mean


def train_cycle(epoch, writer):
    running_loss = 0.0
    last_loss = 0.0

    # Train loop
    for i, batch in enumerate(train_loader):
        inputs, actions, labels = batch

        actions = actions.float()
        labels = labels.float()

        inputs = inputs.to(device)
        actions = actions.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs, actions)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.view(-1), labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        running_loss += loss.item()
        if i % 20 == 19:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(train_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

         # Print model's output, target values, and loss for a few samples
        #if i % 100 == 0:  # Adjust this value as needed to print less or more frequently
            #print(f"Batch: {i}, Output: {outputs.view(-1)[:5]}, Target: {labels[:5]}, Loss: {loss.item()}")

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/res_rot_dist_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10000

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    avg_loss = train_cycle(epoch_number, writer)

    # We don't need gradients on to do reporting
    net.train(False)

    running_vloss = 0.0

    for i, vdata in enumerate(val_loader):
        # Extract the inputs and labels from the sample
        vinputs, vactions, vlabels = vdata
        vactions = vactions.float()
        vlabels = vlabels.float()

        # Forward pass
        vinputs = vinputs.to(device)
        vactions = vactions.to(device)
        vlabels = vlabels.to(device)

        voutputs = net(vinputs, vactions)
        vloss = loss_fn(voutputs.view(-1), vlabels)
        running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    
    print('train loss: {} valid loss: {}'.format(avg_loss, avg_vloss))
    print()

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'res_rot_dist_{}_{}'.format(timestamp, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1