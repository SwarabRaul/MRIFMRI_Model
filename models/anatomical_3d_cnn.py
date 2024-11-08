import torch
import torch.nn as nn
import torch.nn.functional as F

class Anatomical3DCNN(nn.Module):
    def __init__(self):
        super(Anatomical3DCNN, self).__init__()
        
        # First 3D Convolutional layer
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Second 3D Convolutional layer
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Third 3D Convolutional layer
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16 * 16, 256)  # Adjust dimensions based on your input size
        self.fc2 = nn.Linear(256, 1)  # Binary classification output

    def forward(self, x):
        # Pass through convolutional layers with ReLU and pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 16 * 16 * 16)  # Adjust dimensions based on your input size
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x
