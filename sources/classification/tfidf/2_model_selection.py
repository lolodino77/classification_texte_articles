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
set_current_directory_to_root(root = "classification_texte_bapteme_philo", filetype = "notebook")
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
    files_to_open = sys.argv[1] # argument du script, si files_to_open==in_command execute le script sur les 
    # fichiers (datasets) entres en arguments dans la ligne de commande, 
    # mais si files_to_open==in_input_repertory execute le script sur tous les fichiers du dossier ./data/input
    
    files_format = sys.argv[2] # format des fichiers datasets a ouvrir (parquet, csv, etc.), multiple si plusieurs formats
    # sert quand files_to_open==in_input_repertory, pour n'importer que les parquet, ou que les csv, etc.

    if(files_to_open == "in_command"):
        if(len(sys.argv) == 3): # cas quand il n'y a qu'un seul dataset => il faut creer une liste
            filenames = [sys.arg[3]]
        else: #cas quand il y a au moins deux datasets => pas besoin de creer de liste
            filenames = sys.argv[3:] # ignorer les 2 premiers arguments, le nom du script et files_to_open
    elif(files_to_open == "in_input_repertory"):
        filenames = glob.glob(os.path.join(os.getcwd() + "\\data\\input", "*." + files_format))
        filenames = [filename.split("input\\")[1] for filename in filenames] # enlever le path entier, garder que le nom du fichier

    select_models(filenames)
    
    
main()