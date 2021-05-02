import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import load_huesken_data

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

X_numpy, y_numpy = load_huesken_data(datafile='../../data/HueskenRNA_success.csv', n_samples=1000)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.int))
# y = y.view(y.shape[0], 1)
# print(y.shape)

num_classes = 2
n_samples, n_features = X.shape

# Model
input_size = n_features
hidden_size = 100
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
learning_rate = 0.003
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = n_samples
loss_array = []
for epoch, (x_step, y_step) in enumerate(zip(X, y)):
    y_predicted = model(x_step)
    with torch.no_grad():
        print('x_step', x_step.shape)
        print('y_step', y_step.shape, y_step)
        print('y_predicted', y_predicted.shape, y_predicted)
    loss = criterion(y_predicted, y_step)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # with torch.no_grad():
    #     print(f'actual: {y_step.item():.6f}, predicted: {y_predicted.item():.6f}, loss: {loss.item():.6f}')
    loss_array.append(loss.item())
    # if (epoch + 1) % 10 == 0:
    #     print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# predicted = model(X_numpy).detach().numpy()
plt.plot(loss_array, 'ro')
# plt.plot(predicted, 'b')
plt.show()
