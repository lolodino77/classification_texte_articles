import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 20)

from pathlib import Path, PureWindowsPath
os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = PureWindowsPath(os.getcwd() + "\\data\\input\\data.parquet")
path = path.as_posix()
corpus = pd.read_parquet(path) #engine="fastparquet"
corpus = corpus.sample(frac=1).reset_index(drop=True)

X = corpus["message_preprocessed"]
y = corpus["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Builds a dictionary of features and transforms documents to feature vectors and convert our text documents to a
# matrix of token counts (CountVectorizer)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# entrainement du modele
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# test du modele
X_test_counts = count_vect.transform(X_test) 
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# on affiche les documents a propos desquels le modele s'est trompe
corpus_test = pd.DataFrame({"message_preprocessed":X_test, "truth":y_test, "pred":y_pred})
corpus_test_errors = corpus_test.query("truth != pred")
corpus_test_errors = corpus_test_errors[["truth", "pred", "message_preprocessed"]]
corpus_test_errors.to_csv("data/output/prediction_errors.csv", index=False)

# on affiche les poids des mots tfidf
idf = tfidf_transformer.idf_
df_idf_weights = pd.DataFrame({"words":count_vect.get_feature_names(), "idf":idf})
df_idf_weights =  df_idf_weights.sort_values("idf", ascending=False)
# print(pd.DataFrame({"words":count_vect.get_feature_names(), "idf":idf}))

idf = tfidf_vectorizer.idf_
scores_tfidf = pd.DataFrame({"words":tfidf_vectorizer.get_feature_names(), "idf":idf})

# Sources :
# https://iq.opengenus.org/text-classification-naive-bayes/
