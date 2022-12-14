import sys
import os
import glob
from pathlib import PureWindowsPath


def set_current_directory_to_root(root):
	"""Se place dans le root

	Parametres: 
	root (string) : Le nom du root (un dossier) auquel on veut se rendre
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


def get_all_files_from_a_directory(path_to_directory, files_extension=""):
	""" Donne la liste de tous les fichiers (du meme format ou non selon ce qu'on veut) d'un dossier 
	Exemple path_to_directory = "./data/input/corpus_txt/"
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
	""" Donne l'extension d'un fichier """
	extension = filename.split(".")[1]
	return(extension)


def get_corpus_name_from_filename(filename):
	""" input : corpus_chat_chien.csv ==> output : chat_chien"""
	corpus_name = filename.split(".")[0].split("corpus_")[1]
	return(corpus_name)


def check_duplicates(data, id_col_name):
	"""Verifie la presence ou non de doublons dans un dataframe

	Parametres: 
	data (pandas DataFrame) : Le dataframe dont on verifie la presence ou non de doublons
	id_col_name (string) : Le nom de la colonne qui contient les id (la cle primaire)
	"""
	print("presence de doublons ?")
	print(data[id_col_name].duplicated().any())
	print(data.index.duplicated().any())
	   

def save_list_to_txt(input_list, path_to_file, file_open_mode, sep):
	""" Ecrit une liste input_list (avec saut a la ligne) dans un fichier texte situe au path path_to_file
	Entrees:
		input_list (liste de string) : La liste de string a ecrire dans le fichier texte
		sep (string) : Le separateur entre deux textes du fichier texte (\n, \n\n, etc.)
	"""
	f = open(path_to_file, file_open_mode, encoding="utf-8") #"w" si n'existe pas, "a" si on veut ajouter a un fichier deja existant
	for line in input_list:
		f.write(line + sep)
	f.close()