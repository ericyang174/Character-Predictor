import string
import json
import torch
from predutils import normalize

class CharDictionary:
    def __init__(self, dictionary=None):
        self.dictionary = dictionary
        self.DICT_FNAME = 'dict.json'

    def fit(self, fpath, odir='./work'):
        self.dictionary = {}

        data = normalize(fpath)

        chrs = set()
        for txt in data:
            print(txt)
            for chr in txt:
                chrs.add(chr)

        for idx, chr in enumerate(sorted(chrs)):
            self.dictionary[chr] = idx + 1

        with open(f'{odir}/{self.DICT_FNAME}', 'w') as fdict:
            fdict.write(json.dumps(self.dictionary))

    def transform_input(self, fpath, wisize=5):
        if self.dictionary is None:
            raise Exception("Need to fit Dictionary before transforming")

        embeddings = []

        input = open(fpath, 'r')
        print(self.dictionary)

        for data in input:
            vec = []
            for chr in data:
                if chr == '\n':
                    continue

                if chr in self.dictionary:
                    vec.append(self.dictionary[chr])
                else:
                    print(len(chr))
                    vec.append(len(self.dictionary) + 1)

            embeddings.append(vec)

        input.close()

        embeddings = torch.tensor(embeddings).reshape(len(embeddings), wisize, 1)
        embeddings = embeddings / (len(self.dictionary) + 2)

        return embeddings

    def transform_output(self, fpath):
        if self.dictionary is None:
            raise Exception("Need to fit Dictionary before transforming")
        
        out_embeddings = []

        output = open(fpath, 'r')
        for chr in output:
            chr = chr.replace('\n', '')

            out_embeddings.append(self.dictionary[chr] - 1)

        return torch.tensor(out_embeddings)