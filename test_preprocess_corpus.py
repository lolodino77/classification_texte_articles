import pandas as pd
import numpy as np
import nltk
import re
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
pd.set_option('display.max_colwidth', 600)
pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

df = pd.read_csv('dataset_philosophy.csv')
sentence = df.iloc[10][0]
sentence_original = sentence
print("sentence")
print(sentence)
print()


language = "french"
french_stopwords = nltk.corpus.stopwords.words(language)
mots = set(line.strip() for line in open('dictionnaire.txt'))
lemmatizer = FrenchLefffLemmatizer()

#remplacer les virgules bizarres
print("avant d'enlever les virgules bizarres")
print(sentence)
sentence = sentence.replace("’", "'")
print(sentence)

#virer les mots avant les apostrophes
sentence = re.sub(r"\s\w+'", " ", sentence, 0)

# enlever la ponctuation et met en minuscule
sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])
print("sentence_w_punct")
print(sentence_w_punct)
print()

# enlever les chiffres
sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())
print("sentence_w_num")
print(sentence_w_num)
print()

# transformer les phrases en liste de tokens (en liste de mots)
tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)
print(tokenize_sentence)
print()

# enlever les stopwords (mots n’apportant pas de sens)
words_w_stopwords = [i for i in tokenize_sentence if i not in french_stopwords]
print("words_w_stopwords")
print(words_w_stopwords)
print()

# lemmatizer
words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords) #words_lemmatize est un iterateur
words_lemmatize = list(words_lemmatize)

# reformer la phrase en reliant les mots precedents
sentence_clean = " ".join(words_lemmatize)

print("sentence_clean =", sentence_clean)
print("sentence_original =", sentence_original)