from datasets import load_dataset
import random

# Load datasets and join the train and validation together
english_dataset = load_dataset('wikimedia/wikipedia', '20231101.en')
chinese_dataset = load_dataset('wikimedia/wikipedia', '20231101.zh')
spanish_dataset = load_dataset('wikimedia/wikipedia', '20231101.es')
hindi_dataset = load_dataset('wikimedia/wikipedia', '20231101.hi')
french_dataset = load_dataset('wikimedia/wikipedia', '20231101.fr')
russian_dataset = load_dataset('wikimedia/wikipedia', '20231101.ru')
arabic_dataset = load_dataset('wikimedia/wikipedia', '20231101.ar')

# Filter out empty lines from the 'text' column
non_empty_text_english = [line.replace("\n", " ") for line in english_dataset["train"]["text"] if line.strip() != ""]
non_empty_text_chinese = [line.replace("\n", " ") for line in chinese_dataset["train"]["text"] if line.strip() != ""]
non_empty_text_spanish = [line.replace("\n", " ") for line in spanish_dataset["train"]["text"] if line.strip() != ""]
non_empty_text_hindi = [line.replace("\n", " ") for line in hindi_dataset["train"]["text"] if line.strip() != ""]
non_empty_text_french = [line.replace("\n", " ") for line in french_dataset["train"]["text"] if line.strip() != ""]
non_empty_text_russian = [line.replace("\n", " ") for line in russian_dataset["train"]["text"] if line.strip() != ""]
non_empty_text_arabic = [line.replace("\n", " ") for line in arabic_dataset["train"]["text"] if line.strip() != ""]

# Sample articles for each language
random.seed(42)

random.shuffle(non_empty_text_english)
english_sample = non_empty_text_english[:30000]

random.shuffle(non_empty_text_chinese)
chinese_sample = non_empty_text_chinese[:15000]

random.shuffle(non_empty_text_spanish)
spanish_sample = non_empty_text_spanish[:15000]

random.shuffle(non_empty_text_hindi)
hindi_sample = non_empty_text_hindi[:10000]

random.shuffle(non_empty_text_french)
french_sample = non_empty_text_french[:10000]

random.shuffle(non_empty_text_russian)
russian_sample = non_empty_text_russian[:10000]

random.shuffle(non_empty_text_arabic)
arabic_sample = non_empty_text_arabic[:10000]

# Take the first 500-1000 characters for each article
english_sample = [random_length_article(article) for article in english_sample]
chinese_sample = [random_length_article(article) for article in chinese_sample]
spanish_sample = [random_length_article(article) for article in spanish_sample]
hindi_sample = [random_length_article(article) for article in hindi_sample]
french_sample = [random_length_article(article) for article in french_sample]
russian_sample = [random_length_article(article) for article in russian_sample]
arabic_sample = [random_length_article(article) for article in arabic_sample]

# Join the lines into a single string
text_str_english = "\n".join(english_sample)
text_str_chinese = "\n".join(chinese_sample)
text_str_spanish = "\n".join(spanish_sample)
text_str_hindi = "\n".join(hindi_sample)
text_str_french = "\n".join(french_sample)
text_str_russian = "\n".join(russian_sample)
text_str_arabic = "\n".join(arabic_sample)

with open("data/wikitext_train.txt", "w") as f:
    f.write(text_str_english)
    f.write(text_str_chinese)
    f.write(text_str_spanish)
    f.write(text_str_hindi)
    f.write(text_str_french)
    f.write(text_str_russian)
    f.write(text_str_arabic)

def random_length_article(article):
    # Generate a random number between 500 and 1000
    length = random.randint(500, 1000)
    return article[:length]