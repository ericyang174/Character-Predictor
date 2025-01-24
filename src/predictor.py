import argparse
import yaml
import os
import numpy as np
import string
import torch
import torch.nn as nn

class CharPredictor(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x

def build_char_dictionary(f_path):
    normalized_lines = []
    unique_char = set()
    with open(f_path, 'r') as f:
        for line in f:
            line = line.lower()
            line = ''.join([char for char in line if char not in string.punctuation])
            line = line.strip()
            for char in line:
                unique_char.add(char)
            normalized_lines.append(line)
    
    sorted_char = sorted(list(unique_char))
    char_dictionary = {}
    for i in range(len(sorted_char)):
        char_dictionary[sorted_char[i]] = i + 1

    return normalized_lines, char_dictionary

def parse_train_data(normalized_lines, window_size, char_dictionary):
    X_train = []
    y_train = []

    for line in normalized_lines:
        for i in range(0, len(line) - window_size):
            input = line[i : i + window_size]
            output = line[i + window_size]
            char_vec = []
            for char in input:
                char_vec.append(char_dictionary[char])
            X_train.append(char_vec)
            y_train.append(char_dictionary[output] - 1)
    
    X_train = torch.tensor(X_train).reshape(len(X_train), window_size, 1)

    # Includes + 2 because 0 is saved for padding and number (dictionary + 1) represents
    # unseen values when evaluating model
    X_train = X_train / (len(char_dictionary) + 2)
    y_train = torch.tensor(y_train)

    return X_train, y_train

def train(config, X_train, y_train, predictor):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=config["batch_size"])
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=config["batch_size"])
    # data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), 
    #                                 shuffle=True, batch_size=config["batch_size"])
    
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
        
        # TODO: Split data into train and validation
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
                argmax = torch.argmax(y_pred, dim=1)
                for i in range(len(argmax)):
                    if (argmax[i] == y_batch[i]):
                        correct += 1
                        print(argmax[i])
                total_correct += correct

            print(f"Epoch: {epoch} Loss: {total_loss / total_sample} Correct: {total_correct} out of {total_sample}")
    
    torch.save([predictor], "work/model.pth")

def main():
    parser = argparse.ArgumentParser(description="Trains or predicts characters with model")
    parser.add_argument('-c', '--config', required=False, help="Path to configuation")
    parser.add_argument('-m', '--mode', required=True, help="Mode should be either train or test")
    parser.add_argument('-w', '--work_dir', required=False, help="Path to save model checkpoints")
    parser.add_argument('--test_data', required=False, help="Path to the test data")
    parser.add_argument('--test_output', required=False, help="Path to save output")

    args = parser.parse_args()

    mode = args.mode 

    if mode == 'train':
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

    if mode == 'train':
        # if not os.path.isdir(args.work_dir):
        #     print('Making working directory {}'.format(args.work_dir))
        #     os.makedirs(args.work_dir)

        normalized_lines, char_dictionary = build_char_dictionary("./data/dummy_train.txt")
        X_train, y_train = parse_train_data(normalized_lines, config["window_size"], char_dictionary)
        predictor = CharPredictor(hidden_size=config["hidden_size"], vocab_size=len(char_dictionary))
        train(config, X_train, y_train, predictor)
        print(char_dictionary)
        
    elif mode == 'test':
        print("Loading model...")
        # LOAD MODEL
        print("Loading test data...")
        # LOAD TEST DATA
        print("Making predictions")
        # PREDICT AND WRITE TO FILE
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

if __name__ == '__main__':
    main()