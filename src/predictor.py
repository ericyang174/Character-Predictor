import argparse
import yaml
import os
import numpy as np
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

def main():
    parser = argparse.ArgumentParser(description="Trains or predicts characters with model")
    parser.add_argument('-c', '--config', required=True, help="Path to configuation")
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
        predictor = CharPredictor(64, 10)
        print("Loading training data...")
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