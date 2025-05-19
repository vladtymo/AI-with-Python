import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize

text = "This is an example sentence, showing off the tokenization process."

tokens = word_tokenize(text)


print(tokens)

# ['This', 'is', 'an', 'example', 'sentence', ',', 'showing', 'off', 'the', 'tokenization', 'process', '.']
