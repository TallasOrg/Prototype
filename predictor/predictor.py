import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import load_huesken_data

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

X_numpy, y_numpy = load_huesken_data(datafile='../data/HueskenRNA_success.csv', n_samples=1000)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# Model
input_size = n_features
output_size = 1
model = NeuralNet(input_size, output_size)

# loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = n_samples
loss_array = []
for epoch, (x_step, y_step) in enumerate(zip(X, y)):
    y_predicted = model(x_step)
    loss = criterion(y_predicted, y_step)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        print(f'actual: {y_step.item():.6f}, predicted: {y_predicted.item():.6f}, loss: {loss.item():.6f}')
    loss_array.append(loss.item())
    # if (epoch + 1) % 10 == 0:
    #     print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# predicted = model(X_numpy).detach().numpy()
plt.plot(loss_array, 'ro')
# plt.plot(predicted, 'b')
plt.show()
