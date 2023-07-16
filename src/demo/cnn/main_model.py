import torch
import torch.nn as nn
import torch.optim as optim

class CnnChordClassifier(nn.Module):
    def __init__(self, num_classes, hidden_layers):
        super(CnnChordClassifier, self).__init__()
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        
        self.drop = nn.Dropout(p=0.3)
        
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        
        self.conv1 = nn.Conv2d(3, self.hidden_layers[0], kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(self.hidden_layers[0])
        
        self.conv2 = nn.Conv2d(self.hidden_layers[0], self.hidden_layers[1], kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(self.hidden_layers[1])
        
        self.conv3 = nn.Conv2d(self.hidden_layers[1], self.hidden_layers[2], kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(self.hidden_layers[2])
        
        self.conv4 = nn.Conv2d(self.hidden_layers[2], self.hidden_layers[3], kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(self.hidden_layers[3])
        
        self.conv5 = nn.Conv2d(self.hidden_layers[3], self.hidden_layers[4], kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(self.hidden_layers[4])
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.hidden_layers[5]*4, self.num_classes)

    def forward(self, x):
        # Layer 1
        out1 = self.conv1(x)
        out1 = self.batchnorm1(out1)
        out1 = self.maxpool(out1)
        out1 = self.relu(out1)
        out1 = self.drop(out1)

        # Layer 2
        out2 = self.conv2(out1)
        out2 = self.batchnorm2(out2)
        out2 = self.maxpool(out2)
        out2 = self.relu(out2)
        out2 = self.drop(out2)

        # Layer 3
        out3 = self.conv3(out2)
        out3 = self.batchnorm3(out3)
        out3 = self.maxpool(out3)
        out3 = self.relu(out3)
        out3 = self.drop(out3)

        # Layer 4
        out4 = self.conv4(out3)
        out4 = self.batchnorm4(out4)
        out4 = self.maxpool(out4)
        out4 = self.relu(out4)
        out4 = self.drop(out4)

        # Layer 5
        out5 = self.conv5(out4)
        out5 = self.batchnorm5(out5)
        out5 = self.maxpool(out5)
        out5 = self.relu(out5)
        out5 = self.drop(out5)

        # Flatten and fully connected layer
        out5 = self.flatten(out5)
        out = self.linear(out5)
        return out
