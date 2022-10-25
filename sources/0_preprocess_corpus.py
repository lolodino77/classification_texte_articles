import pandas as pd
import numpy as np
import nltk
import re
import string
import os
from pathlib import Path, PureWindowsPath
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 20)

def preprocess_list_of_documents(listofdocuments, lemmatizer):
# cas speciaux a traiter
# mots avec un apostrophe avant (Traite)
# mots composes avec un ou plusieurs tirets (A traiter)
	preprocess_list = []
	for document in listofdocuments :
		#remplacer les virgules bizarres
		document = document.replace("’", "'")

		# supprimer les mots avant les apostrophes (particules comme l', t', etc.)
		document = re.sub(r"\s\w+'", " ", document, 0)

		# enlever la ponctuation et met en minuscule
		ponctuation_to_remove = string.punctuation.replace("-", "")
		document_w_punct = "".join([i.lower() for i in document if i not in ponctuation_to_remove])

		# enlever les chiffres
		document_w_num = ''.join(i for i in document_w_punct if not i.isdigit())

		# transformer les phrases en liste de tokens (en liste de mots)
		tokenize_document = nltk.tokenize.word_tokenize(document_w_num)

		# enlever les stopwords (mots n’apportant pas de sens)
		words_w_stopwords = [i for i in tokenize_document if i not in french_stopwords]

		# lemmatizer (convertir en la racine)
		words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords) #words_lemmatize est un iterateur
		words_lemmatize = list(words_lemmatize)

		# reformer la phrase en reliant les mots precedents
		document_clean = " ".join(words_lemmatize)

		#rajouter la phrase dans la liste
		preprocess_list.append(document_clean)
	return preprocess_list

#Pas besoin si tout est deja installe
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

print(os.getcwd())
os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..')))
print(os.getcwd())
corpus_philosophy = pd.read_csv('data/input/dataset_philosophy.csv')
corpus_baptism = pd.read_csv('data/input/dataset_baptism.csv')

corpus_philosophy["category"] = "philosophy"
corpus_baptism["category"] = "baptism"
	
corpus = pd.concat([corpus_philosophy, corpus_baptism]) 
print(corpus.shape)
print(corpus.columns)

language = "french"
french_stopwords = nltk.corpus.stopwords.words(language)
print("os.getcwd() =", os.getcwd())
mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))

lemmatizer = FrenchLefffLemmatizer()
corpus["message_preprocessed"] = preprocess_list_of_documents(corpus['message'], lemmatizer)

corpus.index = list(range(len(corpus)))
corpus["id"] = corpus.index	
#corpus["id"] = list(range(len(corpus)))

corpus = corpus[["id", "message", "message_preprocessed", "category"]]
corpus["length"] = corpus["message"].str.len()

corpus["category_bin"] = np.select([corpus["category"] == "philosophy"], [1], default=0)
corpus = corpus.sample(frac=1).reset_index(drop=True)

#enlever les retours a la ligne \n et \r
corpus.replace("\\n", " ", regex=True, inplace=True)
corpus.replace("\\r", " ", regex=True, inplace=True)

#supprimer les doublons
print("corpus.shape =", corpus.shape)
corpus.drop_duplicates("message", inplace=True, keep="first")
print("corpus.shape =", corpus.shape)

#pour enlever les faux exemples : commentaires, description auteur, texte anglais, references bibliographiques

# Enregistrer le corpus
path = PureWindowsPath(os.getcwd() + "\\data\\input\\data.parquet")
path = path.as_posix()
corpus.to_parquet(path, engine="fastparquet")
corpus = pd.read_parquet(path) #engine="fastparquet"
print(corpus)

#Credit source : 
#https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/