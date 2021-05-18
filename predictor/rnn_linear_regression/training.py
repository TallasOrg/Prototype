import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import HueskenDataset, ToTensor, RNA_alphabet
from model import rnnLinearRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = HueskenDataset(datafile='../../data/HueskenRNA_success.csv', transform=ToTensor())

for hidden_dim in [10, 20, 30]:
    for n_layers in [1, 2, 3]:
        input_size = len(RNA_alphabet) + 1
        num_epochs = 5
        batch_size = 1
        learning_rate = 0.001
        optimizer_type = 'adam'

        LOG_NAME = f'{hidden_dim}-hidden_dim-{n_layers}-n_layers-{learning_rate}-lr-{optimizer_type}-optim-{int(time.time())}'
        writer = SummaryWriter(f'logs/{LOG_NAME}')

        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        examples = iter(loader)
        example_features, example_labels = examples.next()

        model = rnnLinearRegression(input_size, hidden_dim, n_layers)
        criterion = nn.MSELoss()
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        writer.add_graph(model, torch.squeeze(example_features, 0))

        n_total_steps = len(loader)
        running_loss = 0.0
        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(loader):
                features = torch.squeeze(features, 0)
                labels = labels.to(device)
                features = features.to(device)

                outputs, hidden = model(features)
                loss = criterion(outputs[-1,:], labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

                if (i+1) % 50 == 0:
                    print(f'hidden_dim {hidden_dim}, n_layers {n_layers}, epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
                    writer.add_scalar('training loss', running_loss / 50, epoch * n_total_steps + i)
                    running_loss = 0.0
        writer.close()