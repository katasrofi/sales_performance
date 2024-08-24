import torch
from torch import nn

class LSTMNN(nn.Module):
    def __init__(self,
                 input_layer,
                 hidden_layer,
                 sequence_layer,
                 output_layer):
        super(LSTMNN, self).__init__()
        self.layer1 = nn.Linear(input_layer,
                                hidden_layer)
        self.lstm = nn.LSTM(hidden_layer,
                            hidden_layer,
                            num_layers=sequence_layer,
                            batch_first=True)
        self.relu1 = nn.ReLU()
        self.output = nn.Linear(hidden_layer,
                                output_layer)

    def forward(self, x):
        y = self.layer1(x)
        y, _ = self.lstm(y)
        y = self.relu1(y)
        y = self.output(y)
        return y
