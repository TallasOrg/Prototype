import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import HueskenDataset, ToTensor, smiles_alphabet, collate_fn
from model import lstmLinearRegression
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = HueskenDataset(datafile='../../../data/HueskenRNA_success.csv', transform=ToTensor())
training_length = int(0.7 * len(dataset))
val_length = len(dataset) - training_length
training_dataset, val_dataset = torch.utils.data.random_split(dataset, [training_length, val_length], generator=torch.Generator().manual_seed(42))

for hidden_dim in [10]:
    for n_layers in [1]:
        for learning_rate in [0.001]:
            for optimizer_type in ['adam']:
                for criterion_type in ['mse']:
                    input_size = len(smiles_alphabet) + 1
                    num_epochs = 3
                    batch_size = 50

                    model = lstmLinearRegression(input_size, hidden_dim, n_layers)
                    if criterion_type == 'mse':
                        criterion = nn.MSELoss()
                    else:
                        criterion = nn.L1Loss()
                    if optimizer_type == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    else:
                        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                    
                    training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
                    
                    # Tensorboard Setup
                    DIR_NAME = f'{hidden_dim}-hidden_dim-{n_layers}-n_layers-{learning_rate}-lr-{optimizer_type}-optim-{criterion_type}-criterion-{int(time.time())}'
                    training_writer = SummaryWriter(f'logs/{DIR_NAME}/training')
                    validation_writer = SummaryWriter(f'logs/{DIR_NAME}/validation')
                    examples = iter(training_loader)
                    example_features, example_labels = examples.next()
                    training_writer.add_graph(model, example_features, 0)
                    validation_writer.add_graph(model, example_features, 0)

                    print(f'Params: hidden dim - {hidden_dim}, n layer - {n_layers}, learning rate - {learning_rate}, optimizer - {optimizer_type}, criterion - {criterion_type}')
                    for epoch in range(num_epochs):
                        print(f'Epoch {epoch + 1}')
                        
                        # Training
                        n_total_steps = len(training_loader)
                        running_training_loss = 0.0
                        with tqdm(training_loader, unit='batch') as tepoch:
                            for i, (features, labels) in enumerate(tepoch):
                                tepoch.set_description(f'Training')
                                features.to(device)
                                labels = labels.to(device)
                                outputs = model(features)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()

                                tepoch.set_postfix(loss=f'{loss.item():.4f}')
                                training_writer.add_scalar('loss', loss.item(), epoch * n_total_steps + i)
                        
                        # Validation
                        n_total_steps = len(val_loader)
                        running_val_loss = 0.0
                        with tqdm(val_loader, unit='batch') as tepoch:
                            with torch.no_grad():
                                n_total_steps = len(val_loader)
                                running_val_loss = 0.0
                                for i, (features, labels) in enumerate(tepoch):
                                    tepoch.set_description(f'Validation')
                                    features = features.to(device)
                                    labels = labels.to(device)
                                    outputs = model(features)
                                    loss = criterion(outputs, labels)

                                    tepoch.set_postfix(loss=f'{loss.item():.4f}')
                                    validation_writer.add_scalar('loss', loss.item(), epoch * n_total_steps + i)

                    training_writer.close()
                    validation_writer.close()