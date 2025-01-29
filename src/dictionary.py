import string
import yaml
import torch
from predutils import normalize

class CharDictionary:
    def __init__(self, dictionary=None):
        self.dictionary = dictionary
        self.DICT_FNAME = 'dict.yml'

    def get_dict(self):
        return self.dictionary

    '''
    Produces the character dictionary used to encode/decode characters

        fpath: path to the input data
        out_dir: directory path where dictionary is saved
    '''
    def fit(self, fpath, out_dir='work'):
        self.dictionary = {}

        data = normalize(fpath)

        chars = set()
        for txt in data:
            for char in txt:
                chars.add(char)

        for idx, char in enumerate(sorted(chars)):
            self.dictionary[char] = idx + 1

        with open(f'{out_dir}/{self.DICT_FNAME}', 'w') as fdict:
            fdict.write(yaml.dump(self.dictionary))

    '''
    Takes a file path that has training inputs already chunked into window size # of characters per line
    and produces the embeddings, which are normalized between [0, 1)

        fpath: file path to the chunked input training data
        window_size: number of characters per line in the input file
    
    Returns a tensor of shape (# of lines in file, window size, 1) containing the per character embeds
    '''
    def transform_input(self, fpath, window_size=5):
        if self.dictionary is None:
            raise Exception("Need to fit Dictionary before transforming")

        embeddings = []

        input = open(fpath, 'r')

        for data in input:
            pad_count = data.count('.')
            embedding = [0] * pad_count
            for char in data[pad_count:]:
                if char == '\n':
                    continue

                if char in self.dictionary:
                    embedding.append(self.dictionary[char])
                else:
                    embedding.append(len(self.dictionary) + 1) # Considering 0 instead

            embeddings.append(embedding)

        input.close()

        embeddings = torch.tensor(embeddings).reshape(len(embeddings), window_size, 1)
        embeddings = embeddings / (len(self.dictionary) + 2) # If 0 is used, changed to +1 instead of +2

        return embeddings

    '''
    Takes a file with training outputs and returns the one character output as an embedding

        fpath: file path to the output training data
    
    Returns a tensor of embeddings with size (# of lines in the file,)
    '''
    def transform_output(self, fpath):
        if self.dictionary is None:
            raise Exception("Need to fit Dictionary before transforming")
        
        out_embeddings = []

        output = open(fpath, 'r')
        for char in output:
            char = char.replace('\n', '')

            out_embeddings.append(self.dictionary[char] - 1)

        return torch.tensor(out_embeddings)
    
    '''
    Takes any unseen test input data and returns embeddings for the last window size # of 
    characters if possible. If the input data is too short for the window size, paddings
    of zeroes are added to the beginning of the embedding.
    
        fpath: path to the file containing test data
        window_size: size of the window that the model was trained on, which will be applied to test data
    
    Returns a tensor of embeddings with size (# of lines in test file, window size, 1)
    '''
    def transform_test_input(self, fpath, window_size=5):
        if self.dictionary is None:
            raise Exception("Need to fit Dictionary before transforming")

        embeddings = []

        input = open(fpath, 'r')

        for data in input:
            embedding = []
            data = data.replace('\n', '')
            if len(data) < window_size:
                embedding = [0] * (window_size - len(data))

            for char in data[max(0, len(data) - window_size):]:
                if char == '\n':
                    continue

                if char in self.dictionary:
                    embedding.append(self.dictionary[char])
                else:
                    embedding.append(len(self.dictionary) + 1)

            embeddings.append(embedding)

        input.close()

        embeddings = torch.tensor(embeddings).reshape(len(embeddings), window_size, 1)
        embeddings = embeddings / (len(self.dictionary) + 2)

        return embeddings