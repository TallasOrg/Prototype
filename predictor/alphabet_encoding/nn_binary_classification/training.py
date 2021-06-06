import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import HueskenDataset, ToTensor
from model import nnBinaryClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = HueskenDataset(datafile='../../../data/HueskenRNA_success.csv', transform=ToTensor())

hidden_layers = [1, 2, 3]
hidden_sizes = [22, 44, 88]
for num_hidden_layers in hidden_layers:
    for hidden_size in hidden_sizes:
        input_size = 22
        num_epochs = 5
        batch_size = 1
        learning_rate = 0.001
        optimizer_type = 'adam'

        LOG_NAME = f'{hidden_size}-nodes-{num_hidden_layers}-dense-{learning_rate}-lr-{optimizer_type}-optim-{int(time.time())}'
        writer = SummaryWriter(f'logs/{LOG_NAME}')

        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        examples = iter(loader)
        example_features, example_labels = examples.next()

        model = nnBinaryClassification(input_size, hidden_size, num_hidden_layers)
        criterion = nn.BCELoss()
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        writer.add_graph(model, example_features)

        n_total_steps = len(loader)
        running_loss = 0.0
        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(loader):
                labels = labels.to(device)
                features = features.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

                if (i+1) % 50 == 0:
                    print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
                    writer.add_scalar('training loss', running_loss / 50, epoch * n_total_steps + i)
                    running_loss = 0.0
        writer.close()