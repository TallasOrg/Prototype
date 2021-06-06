import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import HueskenDataset, ToTensor, smiles_alphabet
from model import rnnLinearRegression
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = HueskenDataset(datafile='../../../data/HueskenRNA_success.csv', transform=ToTensor())
training_length = int(0.7 * len(dataset))
training_dataset, val_dataset = torch.utils.data.random_split(dataset, [training_length, len(dataset) - training_length], generator=torch.Generator().manual_seed(42))

for hidden_dim in [10, 20, 30]:
    for n_layers in [1, 2, 3]:
        for learning_rate in [0.001, 0.003, 0.01, 0.03]:
            for optimizer_type in ['adam', 'sgd']:
                for criterion_type in ['mse', 'l1']:
                    input_size = len(smiles_alphabet) + 1
                    num_epochs = 3
                    batch_size = 1

                    model = rnnLinearRegression(input_size, hidden_dim, n_layers)
                    if criterion_type == 'mse':
                        criterion = nn.MSELoss()
                    else:
                        criterion = nn.L1Loss()
                    if optimizer_type == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    else:
                        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                    
                    training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
                    
                    # Tensorboard Setup
                    DIR_NAME = f'{hidden_dim}-hidden_dim-{n_layers}-n_layers-{learning_rate}-lr-{optimizer_type}-optim-{criterion_type}-criterion-{int(time.time())}'
                    training_writer = SummaryWriter(f'logs/{DIR_NAME}/training')
                    validation_writer = SummaryWriter(f'logs/{DIR_NAME}/validation')
                    examples = iter(training_loader)
                    example_features, example_labels = examples.next()
                    training_writer.add_graph(model, torch.squeeze(example_features, 0))
                    validation_writer.add_graph(model, torch.squeeze(example_features, 0))

                    print(f'Params: hidden dim - {hidden_dim}, n layer - {n_layers}, learning rate - {learning_rate}, optimizer - {optimizer_type}, criterion - {criterion_type}')
                    for epoch in range(num_epochs):
                        print(f'Epoch {epoch + 1}')
                        
                        # Training
                        n_total_steps = len(training_loader)
                        running_training_loss = 0.0
                        with tqdm(training_loader, unit='batch') as tepoch:
                            for i, (features, labels) in enumerate(tepoch):
                                tepoch.set_description(f'Training')

                                features = torch.squeeze(features, 0)
                                labels = labels.to(device)
                                features = features.to(device)

                                outputs, hidden = model(features)
                                loss = criterion(outputs[-1,:], labels)
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()

                                running_training_loss += loss.item()
                                if (i + 1) % 50 == 0:
                                    running_training_loss = running_training_loss / 50
                                    tepoch.set_postfix(running_loss=f'{running_training_loss:.4f}')
                                    training_writer.add_scalar('loss', running_training_loss, epoch * n_total_steps + i)
                                    running_training_loss = 0.0
                        
                        # Validation
                        n_total_steps = len(val_loader)
                        running_val_loss = 0.0
                        with tqdm(val_loader, unit='batch') as tepoch:
                            with torch.no_grad():
                                n_total_steps = len(val_loader)
                                running_val_loss = 0.0
                                for i, (features, labels) in enumerate(tepoch):
                                    tepoch.set_description(f'Validation')
                                    features = torch.squeeze(features, 0)
                                    labels = labels.to(device)
                                    features = features.to(device)
                                    outputs, hidden = model(features)
                                    loss = criterion(outputs[-1,:], labels)

                                    running_val_loss += loss.item()
                                    if (i + 1) % 20 == 0:
                                        running_val_loss = running_val_loss / 20
                                        tepoch.set_postfix(running_loss=f'{running_val_loss:.4f}')
                                        validation_writer.add_scalar('loss', running_val_loss, epoch * n_total_steps + i)
                                        running_val_loss = 0.0

                    training_writer.close()
                    validation_writer.close()