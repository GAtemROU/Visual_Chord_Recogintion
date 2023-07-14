import torch
import torch.nn as nn
import torch.optim as optim

class CnnChordClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CnnChordClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 11 * 11, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        skip1 = x.copy()
        x = self.conv1(x)
        x = self.relu1(x)
        skip2 = self.maxpool1(x) + skip1
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x) + skip2
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
