import argparse
import yaml
import os
import numpy as np
import string
import torch
import torch.nn as nn
import torch.utils.data as data
import random

class CharPredictor(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x
    
    def __call__(self, x):
        return self.forward(x)

def train(config, X_train, y_train, predictor):
    # Split into train/validation sets
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = data.random_split(dataset, [train_size, val_size])
    train_loader = data.DataLoader(train_data, shuffle=True, batch_size=config["batch_size"])
    val_loader = data.DataLoader(val_data, shuffle=False, batch_size=config["batch_size"])
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=config["lr"])
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        predictor.train()
        for X_batch, y_batch in train_loader:
            y_pred = predictor(X_batch)
            loss = ce_loss(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        predictor.eval()
        total_loss = 0.0
        total_sample = 0.0
        total_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = predictor(X_batch)
                total_loss += len(y_batch) * ce_loss(y_pred, y_batch)
                total_sample += len(y_batch)

                correct = 0
                _, topk_indices = torch.topk(y_pred, k=3, dim=1, largest=True, sorted=True)
                # Check if any of the top 3 predicted labels match the ground truth
                for i in range(len(topk_indices)):
                    if y_batch[i] in topk_indices[i]:
                        correct += 1
                total_correct += correct

            print(f"Epoch: {epoch} Loss: {total_loss / total_sample} Correct: {total_correct} out of {total_sample}")
    
    torch.save(predictor, config["save_path"])