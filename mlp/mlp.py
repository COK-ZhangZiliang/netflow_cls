import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_layers[i]))
            else:
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
