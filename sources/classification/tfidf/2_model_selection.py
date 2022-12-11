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
pd.set_option('display.max_colwidth', None) #afficher texte entier dans dataframesys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources").as_posix())
from pathlib import Path, PureWindowsPath
sys.path.append("../..") # sources
sys.path.append("../") # sources\classification
# sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources").as_posix())
# sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\classification").as_posix())
from lib_general import *
from lib_classification import *


# Se rendre dans le dossier root
set_current_directory_to_root(root = "classification_texte_articles_version_objet")
print("os.getcwd() at root =", os.getcwd()) 

# Ajout des paths necessaires pour importer les librairies perso
# add_paths(paths = ["/sources/classification/"])
# from lib_classification import *


# Les differents cas d'executions :
# python 2_model_selection.py command parquet data_history_baptism.parquet data_philosophy_baptism.parquet
    # ==> Execute le script sur les datasets data_history_baptism.parquet data_philosophy_baptism.parquet
# python 2_model_selection.py command parquet corpus_edwardfeser_exapologist.parquet corpus_alexanderpruss_edwardfeser.parquet
# python 2_model_selection.py command parquet corpus_alexanderpruss_exapologist.parquet
# python 2_model_selection.py ./data/input/merged_corpus/ parquet
    # ==> Execute le script sur tous les datasets parquet dans ./data/input
def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    sys_argv = sys.argv
    format_input = sys_argv[2]
    filenames = get_merged_corpus_filenames(sys_argv)
    print("filenames =", filenames)
    select_models(filenames, format_input)
    
    
main()