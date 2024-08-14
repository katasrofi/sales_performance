import torch
from torch import nn

class NeuralNetworkLinear(nn.Module):
    def __init__(self):
        super(NeuralNetworkLinear, self).__init__()
        self.layer1 = nn.Linear(5, 128)
        self.layer2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, 1)

    def forward(x):
        y = torch.relu(self.layer1(x))
        y = torch.relu(self.layer2(y))
        y = self.output(y)
        return y
