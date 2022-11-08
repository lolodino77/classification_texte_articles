import sys
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.min_rows', 5)
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_colwidth', None) #afficher texte entier dans dataframesys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *

# Se rendre dans le dossier root
set_current_directory_to_root(root = "classification_texte_bapteme_philo", filetype = "notebook")
print("os.getcwd() at root =", os.getcwd()) 

# Ajout des paths necessaires pour importer les librairies perso
add_paths(paths = ["/sources/classification/"])
from lib_classification import *


def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    filename = sys.argv[1]
    # filename = "data_middle_age_epistemology.parquet"
    
    # Initialisation des variables necessaires
    input_or_output = "input"
    # class_col_name = "category"
    id_col_name = "id"
    features_col_names = "message_preprocessed" 
    # class_col_name = "category"
    class_col_name = "category_bin"
    savefig = True

    # Recupere le nom du dataset grace au nom du fichier du dataset filename
    # filename = data_middle_age_epistemology.parquet => dataset_name = middle_age_epistemology
    filename_split = filename.split("data")
    filename_split = filename_split[1].split(".parquet")
    dataset_name = filename_split[0][1:]

    # Importer le dataset puis equilibrer ses classes
    corpus = get_dataset(filename)
    corpus = get_balanced_binary_dataset(corpus, class_col_name)
    print(corpus)

    # Verifier la presence ou non de doublons
    check_duplicates(corpus, id_col_name)

    # Creation du train et du test
    X_train, X_test, y_train, y_test, indices_train, indices_test = get_train_and_test(corpus, features_col_names, class_col_name, id_col_name)
    X_train_tfidf, X_test_tfidf = apply_tfidf_to_train_and_test(X_train, X_test)

    # Creation du dossier de sorties si besoin
    if(savefig):
        os.makedirs("./data/output/" + dataset_name, exist_ok=True)

    # Cross validation
    scorings = ['accuracy', 'f1_macro']
    num_iter = 2 #nombre de repetitions de la k-fold cross validation entiere
    k = 10 #k de la k-fold cross validation
    do_cross_validation(X_train_tfidf, y_train, scorings, num_iter, k, dataset_name)

    Learning curves (du meilleur modele)
    k = 10
    kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=2, random_state=None)
    cv_param = kfold
    num_experiences = 10
    train_sizes = np.linspace(0.1, 1.0, num_experiences)
    n_jobs = -1
    model = SVC()

    scorings = ['accuracy', 'precision']
    get_all_learning_curves(model, X_train_tfidf, y_train, cv_param, scorings, train_sizes, n_jobs=-1, 
                                savefig=savefig, dataset_name=dataset_name)

main()