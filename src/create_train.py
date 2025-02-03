from datasets import load_dataset
import random

# Load datasets and join the train and validation together
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
input_data = " ".join(dataset["train"]["text"]) + " ".join(dataset["validation"]["text"])
input_data = input_data.replace("\n", " ")

start = 0
splits = []
length = len(input_data)

# Create random splits of the data and write them to given file
while start < length:
    end = start + random.randint(20, 1000)
    end = min(end, length)

    splits.append(input_data[start:end])
    start = end

with open("data/wikitext_train.txt", 'w') as f:
    for split in splits:
        f.write(split + "\n")
