import sys
import os
sys.path.insert(0, "..")
print("(3_train...py : sys.path =", sys.path)
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import joblib
from lib_classification import *
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.min_rows', 5)
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_colwidth', None) #afficher texte entier dans dataframe


# Commande terminal :
# python 3_train_test_best_model.py corpus_name
# python 3_train_test_best_model.py corpus_alexanderpruss_edwardfeser.parquet
# python 3_train_test_best_model.py corpus_alexanderpruss_edwardfeser.csv
# python 3_train_test_best_model.py corpus_sceptic_theist.csv
# python 3_train_test_best_model.py corpus_feser_pruss.csv
# python 3_train_test_best_model.py corpus_amazon_raw.parquet

filename_corpus = sys.argv[1]

# Se rendre dans le dossier root
set_current_directory_to_root(root = "classification_texte_articles")
print("os.getcwd() at root =", os.getcwd()) 

# path = PureWindowsPath(os.getcwd() + "/data/input/merged_corpus/{}".format(filename_corpus))
# path = path.as_posix() #convertir en path linux (convertir les \\ en /)
# corpus = pd.read_parquet("./data/input/merged_corpus/{}".format(filename_corpus)) #engine="fastparquet"
# corpus = pd.read_csv("./data/input/merged_corpus/{}".format(filename_corpus)) #engine="fastparquet"

corpus = get_merged_corpus_dataframe_from_filename(filename_corpus)
corpus = get_balanced_binary_dataset(corpus, class_col_name="category")
corpus_name = get_corpus_name_from_filename(filename_corpus)
print("corpus_name =", corpus_name)
le = joblib.load("./data/input/merged_corpus/labelEncoder_category_{}.joblib".format(corpus_name))
print("le =", le)
print("le.class =", le.classes_)
class_names = {"0":le.classes_[0], "1":le.classes_[1]}
print("class_names =", class_names)

print(corpus["category_bin"].value_counts())
print("presence de doublons ?")
print(corpus.id.duplicated().any())
print(corpus.index.duplicated().any())
corpus

model = SGDClassifier()
X = corpus["message_preprocessed"]
y_str = corpus["category"]
y = corpus["category_bin"]
indices = corpus["id"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.33, random_state=42)
print(indices_test)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
model.fit(X_train_tfidf, y_train)

# test du modele
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)
print(y_pred)
print(X_test.shape)
print(X_test_tfidf.shape)

### Obtenir les resultats (sauvegarde sur disque)
# Creation du dossier de sorties si besoin
make_classif_output_dir(corpus_name)

# Evaluation du modele (matrice de confusion, courbe ROC et learning curves)
save_model_diagnostics(corpus, X_train_tfidf, y_train, y_test, y_pred, indices_test, class_names, model, 
                            corpus_name)