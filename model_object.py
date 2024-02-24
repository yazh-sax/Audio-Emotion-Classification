import torch.nn as nn
import torch.nn.functional as f
import torch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 2, 1)
        self.conv2 = nn.Conv2d(4, 8, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 1, 1)
        self.fc1 = nn.Linear(288, 72)
        self.fc2 = nn.Linear(72, 3)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = f.log_softmax(x, dim=1).to(torch.float32)
        return x
