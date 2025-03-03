import argparse
import yaml
import os
import numpy as np
import string
import torch
import torch.nn as nn
import torch.utils.data as data
import random
from tqdm import tqdm
import lightning as pl

class CharPredictor(pl.LightningModule):
    def __init__(self, hidden_size, vocab_size, lr):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_sample = 0.0
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.ce_loss(y_pred, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.ce_loss(y_pred, y)
        self.total_loss += len(y) * loss
        self.total_sample += len(y)

        correct = 0
        _, topk_indices = torch.topk(y_pred, k=3, dim=1, largest=True, sorted=True)
        # Check if any of the top 3 predicted labels match the ground truth
        for i in range(len(topk_indices)):
            if y[i] in topk_indices[i]:
                correct += 1
        self.total_correct += correct
        
        return loss

    def on_validation_epoch_end(self):
        print(f"Epoch: {self.current_epoch} Loss: {self.total_loss / self.total_sample} Correct: {self.total_correct} out of {self.total_sample}")
        self.total_correct = 0.0
        self.total_loss = 0.0
        self.total_sample = 0.0


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, X_train, y_train):
        super().__init__()
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
    
    def setup(self, stage=None):
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        self.train_data, self.val_data = data.random_split(dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return data.DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val_data, shuffle=False, batch_size=self.batch_size)
