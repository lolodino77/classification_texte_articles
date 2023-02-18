import sys
import os
import glob
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
pd.set_option('display.max_colwidth', None) #afficher texte entier dans dataframesys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles\sources").as_posix())
from pathlib import Path, PureWindowsPath
sys.path.append("../..") # sources
sys.path.append("../") # sources\classification
# sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles\sources").as_posix())
# sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles\sources\classification").as_posix())
from lib_general import *
from lib_classification import *


# python 2_model_selection.py corpus_edwardfeser_exapologist.parquet
# python 2_model_selection.py corpus_edwardfeser_exapologist.csv
# python 2_model_selection.py corpus_feser_pruss.csv
# python 2_model_selection.py corpus_amazon.parquet
# python 2_model_selection.py corpus_sceptic_theist.csv
# python 2_model_selection.py
# python 2_model_selection.py all : model selection on all files (csv and parquet)
# python 2_model_selection.py all csv : model selection on all csv files
# python 2_model_selection.py all parquet : model selection on all parquet files

# argv[1] = input files : "all" or "corpus_name.csv" or "corpus_name.parquet"
# argv[2] = input files format : only if argv[1] == "all", equals "csv" or "parquet"

# Se rendre dans le dossier root
set_current_directory_to_root(root = "classification_texte_articles")
print("os.getcwd() at root =", os.getcwd()) 

def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    output = get_merged_corpus_filenames(sys.argv)
    print("output =", output)

    if(len(output) == 2):
        print("len(output) == 2")
        filenames, format_input = output
        select_models_on_multiple_corpus(filenames, format_input)
    elif(len(output) == 1):
        filenames = output
    print("filenames =", filenames)

    # Initialisation des variables necessaires
    id_col_name = "id"
    class_col_name = "category_bin"
    features_col_names = "message_preprocessed"

    for filename in filenames:
        # Recupere le nom du dataset grace au nom du fichier du dataset filename
        print("filename =", filename)
        corpus_name = get_corpus_name_from_filename(filename)
        print("corpus_name =", corpus_name)

        # Creation du dossier de sorties si besoin
        make_classif_output_dir(corpus_name)
        print("make_classif_output_dir(corpus_name)")

        # Importer le dataset puis equilibrer ses classes
        corpus = get_merged_corpus_dataframe_from_filename(filename)
        print("corpus :")
        print(corpus)

        select_models(corpus, corpus_name, id_col_name, class_col_name, features_col_names)
    
        
main()