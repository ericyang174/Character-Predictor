import argparse
import yaml
import random
import string
from dictionary import CharDictionary
import predictor as p
import os
import json
import torch
from predutils import create_train

def main():
    parser = argparse.ArgumentParser(description="Trains or predicts characters with model")
    parser.add_argument('mode', choices=('train, test'), help='What mode to run the program, train or test')
    parser.add_argument('-c', '--config', required=False, default='./src/config.yml', help="Path to configuation")
    parser.add_argument('-w', '--work_dir', required=False, default='./work', help="Path to save/load model checkpoints")
    parser.add_argument('--test_dir', required=False, default='./data', help='Which directory to store the test data and output')
    parser.add_argument('--test_data', required=False, default='test_input.txt', help='Where to store the test data extracted from the dataset')
    parser.add_argument('--test_output', required=False, default='test_output.txt', help="Path to save output")
    parser.add_argument('--dict_path', required=False, default='dict.yml', help='Filename of the dictionary')
    parser.add_argument('--dataset', required=False, default='./data/dummy_train.txt', help="Path to dataset")
    parser.add_argument('--train_dir', required=False, default='./data', help='Which directory to store the training data and output')
    parser.add_argument('--train_input', required=False, default='train_input.txt', help='Where to store the train data extracted from the dataset')
    parser.add_argument('--train_output', required=False, default='train_output.txt', help='Where to store the train output extracted from the dataset')

    args = parser.parse_args()
    mode = args.mode

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if mode == 'train':
        # Creates input/output training data using sliding window
        create_train(args.dataset, config["window_size"])

        # Generates dictionary based on input data
        cdict = CharDictionary()
        cdict.fit(args.dataset)

        # Creates embeddings for input/output training data
        X_train = cdict.transform_input(f'{args.train_dir}/{args.train_input}', config["window_size"])
        y_train = cdict.transform_output(f'{args.train_dir}/{args.train_output}')

        # Creates, trains, and saves model
        model = p.CharPredictor(hidden_size=config["hidden_size"], vocab_size=len(cdict.dictionary))
        p.train(config, X_train, y_train, model)
        
    elif mode == 'test':
        # Opens the file to write predictions to
        foutput = open(f'{args.test_dir}/{args.test_output}', 'w')        

        # Loads in the previously generated dictionary
        with open(f'{args.work_dir}/{args.dict_path}', 'r') as fdict:
            cdict = CharDictionary(yaml.safe_load(fdict))

        # Creates necessary embeddings and loads pretrained model
        X_test = cdict.transform_test_input(f'{args.test_dir}/{args.test_data}', config["window_size"])
        model = torch.load(f'{args.work_dir}/model.pth')[0]

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            _, topk_indices = torch.topk(y_pred, k=3, dim=1, largest=True, sorted=True)

        for i in range(len(topk_indices)):
            # Get top 3 predicted characters for the i-th sample
            dict = cdict.get_dict()
            top_3_indices = topk_indices[i].tolist()
            top_3_chars = [key for idx in top_3_indices for key, value in dict.items() if value == idx + 1]
            
            # Write the top 3 predictions (separated by spaces or commas) for this sample to the file
            foutput.write(''.join(top_3_chars) + '\n')

        foutput.close()
        fdict.close()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

if __name__ == '__main__':
    main()