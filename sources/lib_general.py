import sys
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath


def set_current_directory_to_root(root):
    """Se place dans le root

    Parametres: 
    root (string) : Le nom du root auquel on veut se rendre
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


def get_input_filenames(sys_argv):
    """Obtenir le nom des fichiers de datasets pour l'execution du script 2_model_selection.py

    Parametres: 
    sys_argv (liste de string) : Les arguments de commande pour executer 2_model_selection.py
        Exemples : 
        python 2_model_selection.py /data/input parquet
        python 2_model_selection.py command parquet data_history_baptism.parquet data_philosophy_baptism.parquet
        python create_datasets.py ./data/input/bibliographies/ txt
        python create_datasets.py command txt bibliography_middle_age_fr.txt bibliography_baptism_fr.txt
    Sortie:
    filenames (liste de string) : Le nom des fichiers de datasets pour l'execution du script 2_model_selection.py
    """
    files_to_open = sys_argv[1] # argument du script, si files_to_open==in_command execute le script sur les 
    # fichiers (datasets) entres en arguments dans la ligne de commande, 
    # mais si files_to_open==in_input_repertory execute le script sur tous les fichiers du dossier ./data/input
    
    files_format = sys_argv[2] # format des fichiers datasets a ouvrir (parquet, csv, etc.), multiple si plusieurs formats
    # sert quand files_to_open==in_input_repertory, pour n'importer que les parquet, ou que les csv, etc.
    
    if(files_to_open == "command"):
        if(len(sys_argv) == 3): # cas quand il n'y a qu'un seul dataset => il faut creer une liste
            filenames = [sys_argv[3]]
        else: #cas quand il y a au moins deux datasets => pas besoin de creer de liste
            filenames = sys_argv[3:] # ignorer les 2 premiers arguments, le nom du script et files_to_open
    else:
        input_repertory = files_to_open.replace("/", "\\") # "/data/input/" ==> '\\data\\input\\'
        filenames = glob.glob(os.path.join(input_repertory + "*." + files_format))
        filenames = [filename.split(input_repertory)[1] for filename in filenames] # enlever le path entier, garder que le nom du fichier

    return(filenames)


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
       