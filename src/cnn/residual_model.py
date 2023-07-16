import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += residual  # Skip connection
        out = self.relu2(out)
        return out

class SkipCNN(nn.Module):
    def __init__(self, num_classes):
        super(SkipCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.residual1 = ResidualBlock(64, 64)
        self.residual2 = ResidualBlock(64, 64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 320 * 240, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        return out