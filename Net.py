from config import device
import tcnn
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net1 = nn.Sequential(
        nn.Linear(387, 300),
        nn.LeakyReLU(),
        nn.Linear(300, 128),
        nn.LeakyReLU(),
        nn.Linear(128,16),
        nn.LeakyReLU(),
        nn.Linear(16, 1)

    ).to(device)



    def forward(self, features):
        return self.net1(features)
