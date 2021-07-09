import torch
import torch.nn as nn

class lstmLinearRegression(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers=1, output_size=1, device=None):
        super(lstmLinearRegression, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.device = device
    
    def forward(self, x):
        h0 = self.init_hidden(x.size(0))
        c0 = self.init_hidden(x.size(0))
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    def init_hidden(self, batch_size):
        if self.device:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden