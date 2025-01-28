import string

def create_train(fpath, wisize=5, tpath='./data', Xname='train_input.txt', yname='train_output.txt'):
    Xtrain = open(f'{tpath}/{Xname}', 'w')
    ytrain = open(f'{tpath}/{yname}', 'w')

    norm_txt = normalize(fpath)

    for txt in norm_txt:
        if len(txt) <= wisize:
            txt = ('.' * (wisize - len(txt) + 1)) + txt
        for i in range(len(txt) - wisize):
            input = txt[i : i + wisize]
            output = txt[i + wisize].lower()
            
            Xtrain.write(f'{input}\n')
            ytrain.write(f'{output}\n')

    Xtrain.close()
    ytrain.close()

def normalize(fpath):
    norm_lines = []

    with open(fpath, 'r') as f:
        for line in f:
            line = line.lower()
            line = ''.join([char for char in line if char not in string.punctuation])
            line = line.strip()
            norm_lines.append(line)

    return norm_lines