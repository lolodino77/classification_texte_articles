import sys
import os
import glob
from pathlib import PureWindowsPath
import pandas as pd

""" 
set_current_directory_to_root(root)
add_paths(paths)
get_all_files_from_a_directory(path_to_directory, files_extension="")
get_file_extension(filename)
get_corpus_name_from_filename(filename)
check_duplicates(data, id_col_name)
save_list_to_txt(input_list, path_to_file, file_open_mode, sep)
read_csv(filename, path_to_directory)
read_parquet(filename, path_to_directory)
"""


def set_current_directory_to_root(root):
	""" Place le repertoire courant dans la "racine du projet"

	Entrees : 
		root (string) : Le nom du root (un dossier) auquel on veut se rendre
	"""    
	current_folder = PureWindowsPath(os.path.dirname(os.path.abspath(__file__))).as_posix()
	current_folder_split = current_folder.split(root) # split selon le root
	current_folder_split = current_folder_split[1].split("/")
	dist_to_root = len(current_folder_split) - 1 # nombre de dossier a remonter pour arriver au dossier root
	path_root = "/".join(current_folder.split("/")[:-dist_to_root]) #remonter au dossier root du projet
	os.chdir(path_root)


def add_paths(paths):
	""" Ajoute des paths

	Entrees : 
		paths (liste de string) : Les paths a ajouter
								Exemple : ["/sources/classification/", "/data/], ["/sources/classification/"]
	"""
	for path in paths:
		sys.path.append(os.getcwd() + path)


def get_all_files_from_a_directory(path_to_directory, files_extension=""):
	""" Donne la liste de tous les fichiers (du meme format ou non selon ce qu'on veut) d'un dossier 
	
	Entrees :
		path_to_directory (string) : Le chemin jusqu'au dossier dont on veut lister tous les fichiers
			Exemple : path_to_directory = "./data/input/corpus_txt/"
		files_extension (string) : L'extension des fichiers du dossier qu'on veut garder
			Exemples :
				files_extension = "csv"
				files_extension = "parquet"
	"""
	print("files path_to_directory ", path_to_directory)
	path_to_directory = path_to_directory.replace("/", "\\") # "/data/input/" ==> '\\data\\input\\' format windows

	if(files_extension == ""):
		files_extension = "*"
	else:
		files_extension = "*." + files_extension
	filenames = glob.glob(os.path.join(path_to_directory, files_extension))
	filenames = [filename.split(path_to_directory)[1] for filename in filenames] # enlever le path entier, garder que le nom du fichier

	return(filenames)


def get_file_extension(filename):
	""" Donne l'extension d'un fichier 
	
	Entree :
		filename (string) : Le nom du fichier dont on veut recuperer l'extension
	"""
	extension = filename.split(".")[1]
	return(extension)


def get_corpus_name_from_filename(filename):
	""" Donne le nom d'un corpus a partir de son fichier 
	Exemples : 
		corpus_chat_chien.csv ==> corpus_name : chat_chien
		corpus_mythologie.parquet ==> corpus_name : mythologie

	Entree : 
		filename (string) : Le nom du fichier du corpus dont on veut le nom
			Exemple : corpus_chat_chien.csv
	"""
	print("in get_corpus_name_from_filename")
	print("filename =", filename)
	print("filename.split('.') =", filename.split("."))
	print("filename.split('.')[0] =", filename.split(".")[0])
	corpus_name = filename.split(".")[0].split("corpus_")[1]
	return(corpus_name)


def check_duplicates(data, id_col_name):
	"""Verifie la presence ou non de doublons dans un DataFrame pandas

	Entrees : 
		data (pandas DataFrame) : Le dataframe dont on verifie la presence ou non de doublons
		id_col_name (string) : Le nom de la colonne qui contient les id (la cle primaire)
	"""
	print("presence de doublons ?")
	print(data[id_col_name].duplicated().any())
	print(data.index.duplicated().any())
	   

def save_list_to_txt(input_list, path_to_file, file_open_mode, sep):
	""" Ecrit une liste input_list (avec saut a la ligne) dans un fichier texte situe au path path_to_file
	
	Entrees :
		input_list (liste de string) : La liste de string a ecrire dans le fichier texte
		sep (string) : Le separateur entre deux textes du fichier texte (\n, \n\n, etc.)
	"""
	f = open(path_to_file, file_open_mode, encoding="utf-8") #"w" si n'existe pas, "a" si on veut ajouter a un fichier deja existant
	for line in input_list:
		f.write(line + sep)
	f.close()


def read_csv(filename, path_to_directory):
	""" Lit un fichier .csv """
	df = pd.read_csv(path_to_directory + "/" + filename)
	return(df)


def read_parquet(filename, path_to_directory):
	""" Lit un fichier .parquet """
	df = pd.read_parquet(path_to_directory + "/" + filename)
	return(df)


def get_raw_merged_corpus_from_filename(filename):
    """ Retourne un dataframe pandas a partir d'un fichier de corpus a deux topics (.csv ou .parquet) """
    format = filename.split(".")[1]

    if(format == "csv"):
       df = pd.read_csv("./data/input/raws_merged_corpus/" + filename, encoding="utf-8")
    elif(format == "parquet"):
        df = pd.read_parquet("./data/input/raws_merged_corpus/" + filename)

    return(df)

def get_merged_corpus_dataframe_from_filename(filename):
    """ Retourne un dataframe pandas cree a partir d'un fichier de corpus (.csv ou .parquet) """
    format = filename.split(".")[1]

    if(format == "csv"):
       df = pd.read_csv("./data/input/merged_corpus/" + filename, encoding="utf-8")
    elif(format == "parquet"):
        df = pd.read_parquet("./data/input/merged_corpus/" + filename)

    return(df)