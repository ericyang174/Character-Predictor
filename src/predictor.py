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
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.softmax(self.linear(self.dropout(x)))
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
        char_dictionary[sorted_char[i]] = i

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
            y_train.append(char_dictionary[output])
    
    X_train = torch.tensor(X_train).reshape(len(X_train), window_size, 1)
    X_train = X_train / len(char_dictionary)
    y_train = torch.tensor(y_train)

    return X_train, y_train


def main():
    parser = argparse.ArgumentParser(description="Trains or predicts characters with model")
    parser.add_argument('-c', '--config', required=False, help="Path to configuation")
    parser.add_argument('-m', '--mode', required=True, help="Mode should be either train or test")
    parser.add_argument('-w', '--work_dir', required=False, help="Path to save model checkpoints")
    parser.add_argument('--test_data', required=False, help="Path to the test data")
    parser.add_argument('--test_output', required=False, help="Path to save output")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    mode = args.mode 

    if mode == 'train':
        # if not os.path.isdir(args.work_dir):
        #     print('Making working directory {}'.format(args.work_dir))
        #     os.makedirs(args.work_dir)

        print("Instantiating!")
        predictor = CharPredictor(hidden_size=config["hidden_size"], vocab_size=10)
        print("Loading training data...")
        normalized_lines, char_dictionary = build_char_dictionary("./data/dummy_train.txt")
        X_train, y_train = parse_train_data(normalized_lines, config["window_size"], char_dictionary)
        print(X_train)
        print(y_train)
        print(char_dictionary)
        # LOAD THE DATA
        # TRAIN
        # SAVE
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