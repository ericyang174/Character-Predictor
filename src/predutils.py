import string

'''
Creates training data based on the input data by applying a sliding window, then writes the
input/output to files. Periods are added as padding in case any given line is not bigger than 
the window size.

    fpath: path to the input data to create training data on
    window_size: size of the window applied to each line of the input data
    data_dir: directory in which the new input/output training files will be created
    X_name: the name of the file where training inputs will be written to
    y_name: the name of the file where training outputs will be written to
'''
def create_train(fpath, window_size=5, data_dir='./data', X_name='train_input.txt', y_name='train_output.txt'):
    X_train = open(f'{data_dir}/{X_name}', 'w')
    y_train = open(f'{data_dir}/{y_name}', 'w')

    norm_txt = normalize(fpath)

    for txt in norm_txt:
        if len(txt) <= window_size:
            txt = ('.' * (window_size - len(txt) + 1)) + txt
        for i in range(len(txt) - window_size):
            input = txt[i : i + window_size]
            output = txt[i + window_size]
            
            X_train.write(f'{input}\n')
            y_train.write(f'{output}\n')

    X_train.close()
    y_train.close()

'''
Normalizes input data
  - Lowercase all characters
  - Remove punctuation
  - Removes leading/trailing whitespace and newlines

    fpath: file path to the input data

Returns the normalized data as a list
'''
def normalize(fpath):
    norm_lines = []

    with open(fpath, 'r') as f:
        for line in f:
            line = line.lower()
            line = ''.join([char for char in line if char not in string.punctuation])
            line = line.strip()
            norm_lines.append(line)

    return norm_lines