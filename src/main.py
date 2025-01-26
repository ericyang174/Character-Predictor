import argparse
import yaml
import random
import string
from dictionary import CharDictionary
import predictor as p
import os
import json
from predutils import create_train

def main():
    parser = argparse.ArgumentParser(description="Trains or predicts characters with model")
    parser.add_argument('mode', choices=('train, test'), help='What mode to run the program, train or test')
    parser.add_argument('-c', '--config', required=False, default='./work/config.yml', help="Path to configuation")
    parser.add_argument('-w', '--work_dir', required=False, default='./work', help="Path to save model checkpoints")
    parser.add_argument('--test_data', required=False, help="Path to the test data")
    parser.add_argument('--test_output', required=False, help="Path to save output")
    parser.add_argument('--dict_path', required=False, default='dict.json', help='Filename of the dictionary')
    parser.add_argument('--dataset', required=False, default='./data/dummy_train.txt', help="Path to dataset")
    parser.add_argument('--train_dir', required=False, default='./data', help='Which directory to store the training data and output')
    parser.add_argument('--train_input', required=False, default='train_input.txt', help='Where to store the train data extracted from the dataset')
    parser.add_argument('--train_output', required=False, default='train_output.txt', help='Where to store the train output extracted from the dataset')

    args = parser.parse_args()
    mode = args.mode

    if mode == 'train':
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

        create_train(args.dataset, config["window_size"])

        cdict = CharDictionary()
        cdict.fit(args.dataset)
        X_train = cdict.transform_input(f'{args.train_dir}/{args.train_input}', config["window_size"])
        y_train = cdict.transform_output(f'{args.train_dir}/{args.train_output}')

        model = p.CharPredictor(hidden_size=config["hidden_size"], vocab_size=len(cdict.dictionary))
        p.train(config, X_train, y_train, model)
        print(cdict.dictionary)
        
    elif mode == 'test':
        print("Loading model...")
        # LOAD MODEL
        # if not os.path.isdir(args.dict_dir):
        #     raise Exception('No path for Dictionary')
        # if not os.path.isfile(f'{args.dict_dir}/{args.dict_path}'):
        #     raise Exception('Dictionary JSON doesn\'t exist')
        
        # fdict = open(f'{args.dict_dir}/{args.dict_path}', 'r')
        # fnorm = open(f'{args.dict_dir}/{args.norm_path}', 'r')
        
        # char_dictionary = CharDictionary(json.loads(fnorm), json.loads(fdict))

        # fdict.close()
        # fnorm.close()


        # X_test, y_test = char_dictionary.parse(config["window_size"])
        
        print("Loading test data...")
        # LOAD TEST DATA
        print("Making predictions")
        # PREDICT AND WRITE TO FILE
        finput = open(args.test_data, 'r')
        foutput = open(args.test_output, 'w')

        for _ in finput:
            out = random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + '\n'
            foutput.write(out)

        finput.close()
        foutput.close()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

if __name__ == '__main__':
    main()