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
pd.set_option('display.max_colwidth', None) #afficher texte entier dans dataframesys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *

# Se rendre dans le dossier root
set_current_directory_to_root(root = "classification_texte_bapteme_philo")
print("os.getcwd() at root =", os.getcwd()) 

# Ajout des paths necessaires pour importer les librairies perso
add_paths(paths = ["/sources/classification/"])
from lib_classification import *


# Les differents cas d'executions :
# python 2_model_selection.py in_command parquet data_history_baptism.parquet data_philosophy_baptism.parquet
    # ==> Execute le script sur les datasets data_history_baptism.parquet data_philosophy_baptism.parquet
# python 2_model_selection.py in_input_repertory parquet
    # ==> Execute le script sur tous les datasets parquet dans ./data/input
def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    sys_argv = sys.argv
    filenames = get_intput_filenames(sys_argv)
    select_models(filenames)
    
    
main()