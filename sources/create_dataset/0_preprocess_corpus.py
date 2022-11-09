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


#Pas besoin si tout est deja installe
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

#Lecture des fichiers csv
print(os.getcwd())
os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..' * 2)))
print(os.getcwd())
# corpus_1 = pd.read_csv('data/input/dataset_epistemology.csv')
corpus_1 = pd.read_csv('data/input/dataset_philosophy.csv')
corpus_0 = pd.read_csv('data/input/dataset_baptism.csv')

# Annotation des documents
# corpus_1["category"] = "philosophy"
class_1 = "philosophy"
class_0 = "baptism"
corpus_1["category"] = class_1
corpus_0["category"] = class_0

# Creation du dataset final en regroupant les documents des deux classes
corpus = pd.concat([corpus_1, corpus_0]) 
print(corpus.shape)
print(corpus.columns)

# Recuperation du lemmatizer
language = "french"
french_stopwords = nltk.corpus.stopwords.words(language)
print("os.getcwd() =", os.getcwd())
# mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))
lemmatizer = FrenchLefffLemmatizer()

# Execution de la fonction principale qui fait le nettoyage
corpus["message_preprocessed"] = preprocess_list_of_documents(corpus['message'], lemmatizer)

# Creation de l'id unique
corpus.index = list(range(len(corpus)))
corpus["id"] = corpus.index

# Suppression des colonnes inutiles
corpus = corpus[["id", "message", "message_preprocessed", "category"]]

# Creation de la taille de chaque documents (en nombre de caracteres)
corpus["length"] = corpus["message"].str.len()

# Annotation au format entier (necessaire pour certaines fonctions de sklearn)
corpus["category_bin"] = np.select([corpus["category"] == class_1], [1], default=0)

# Melange aleatoire des documents
corpus = corpus.sample(frac=1).reset_index(drop=True)

# Suppression des retours a la ligne \n et \r
corpus.replace("\\n", " ", regex=True, inplace=True)
corpus.replace("\\r", " ", regex=True, inplace=True)

# Suppression des doublons
print("corpus.shape =", corpus.shape)
corpus.drop_duplicates("message", inplace=True, keep="first")
print("corpus.shape =", corpus.shape)

#pour enlever les faux exemples, preprocessing restant =
#  enlever commentaires, description auteur, texte anglais, references bibliographiques
#  enlever ponctuations (guillemets par exemple) 

# Enregistrer le corpus
path = PureWindowsPath(os.getcwd() + "\\data\\input\\data_" + class_1 + "_" + class_0 + ".parquet")
path = path.as_posix()
corpus.to_parquet(path, engine="fastparquet")
corpus = pd.read_parquet(path) #engine="fastparquet"
print(corpus)

