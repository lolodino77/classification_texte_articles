import sys
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath


def set_current_directory_to_root(root, filetype="notebook"):
    """Se place dans le root

    Parametres: 
    root (string) : Le nom du root auquel on veut se rendre
    filetype (string) : Le type de fichier ou le code est execute, "notebook" ou "source"
    """    
    current_folder = PureWindowsPath(os.path.dirname(os.path.abspath(__file__))).as_posix()
    current_folder_split = current_folder.split(root) # split selon le root
    current_folder_split = current_folder_split[1].split("/")
    dist_to_root = len(current_folder_split) - 1 # nombre de dossier a remonter pour arriver au dossier root
    path_root = "/".join(current_folder.split("/")[:-dist_to_root]) #remonter au dossier root du projet
    os.chdir(path_root)


def add_paths(paths):
    """Equilibre un dataset binaire non equilibre : il aura le meme nombre d'exemples de chaque classe

    Parametres: 
    paths (liste de string) : Les paths a ajouter
                                Exemple : ["/sources/classification/", "/data/], ["/sources/classification/"]
    """
    for path in paths:
        sys.path.append(os.getcwd() + path)


def get_dataset(filename, input_or_output="input"):
    """Equilibre un dataset binaire non equilibre : il aura le meme nombre d'exemples de chaque classe

    Parametres: 
    filename (string) : Le nom du dataframe a importer
                                Exemple : dataset_voitures.parquet, data_velo.csv
    input_or_output (string) : Precise le dossier dans lequel le dataset se trouve, input ou output 

    Sortie:
    data (pandas DataFrame) : Le dataframe recupere a partir du nom de fichier filename
    """
    path = PureWindowsPath(os.getcwd() + "/data/" + input_or_output + "/" + filename) # cree un objet path 
    path = path.as_posix() # convertir en path linux (convertir les \\ en /), renvoie une string
    data = pd.read_parquet(path) #engine="fastparquet"

    return(data)


def check_duplicates(data, id_col_name):
    """Verifie la presence ou non de doublons dans un dataframe

    Parametres: 
    data (pandas DataFrame) : Le dataframe dont on verifie la presence ou non de doublons
    id_col_name (string) : Le nom de la colonne qui contient les id (la cle primaire)
    """
    print("presence de doublons ?")
    print(data[id_col_name].duplicated().any())
    print(data.index.duplicated().any())
       