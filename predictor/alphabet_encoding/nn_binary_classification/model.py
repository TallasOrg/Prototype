import torch
import torch.nn as nn

class nnBinaryClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers=1):
        super(nnBinaryClassification, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation(out)
        for i in range(self.num_hidden_layers - 1):
            out = self.hidden_layer(out)
            out = self.activation(out)
        out = self.output_layer(out)
        out = torch.sigmoid(out)
        return out