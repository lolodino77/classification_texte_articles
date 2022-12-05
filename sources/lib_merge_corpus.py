import nltk
from nltk.stem import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from lib_general import *
from lib_create_corpus import *
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *
import numpy as np
# from lib_classification import *


def merge_two_corpus(corpus_filenames, final_corpus_name, topics, language):
	"""Cree un corpus d'un topic au format pandas dataframe dans le fichier texte filename_output
	Marche pour l'instant que pour fusionner deux corpus (deux topics differents)
	To do : faire pour classification multiclasse

	Parametres: 
	corpus_filenames (liste de string) : Les noms des datasets de corpus a fusionner
					Exemple : ["corpus_philosophy_fr.txt", "corpus_history_fr.txt", "corpus_animals_fr.txt"]
	final_corpus_name (string) : Le nom du fichier dans lequel on ecrira le corpus sous format csv
					Exemple : "dataset_philosophy_history_fr.txt", "dataset_philosophy_history_animals_fr.txt"
	topics (liste de string) : Le nom des topics
					Exemple : ["philosophy", "history"]
	language (string) : La langue des documents
					Valeurs possibles : "french" ou "english"
	
	Sortie:
 	None : Fichier filename_corpus_output qui contient le corpus au format pandas dataframe
	"""
	#Pas besoin si tout est deja installe
	nltk.download('stopwords')
	nltk.download('punkt')
	nltk.download('words')
	nltk.download('wordnet')

	#Lecture des fichiers csv
	# print(os.getcwd())
	# os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..' * 2)))
	# print(os.getcwd())

	# creer version soit csv soit parquet
	corpus_0 = get_corpus_table_from_filename(corpus_filenames[0])
	corpus_1 = get_corpus_table_from_filename(corpus_filenames[1])
	
	# Annotation des documents
	class_0 = topics[0]
	class_1 = topics[1]
	corpus_0["category"] = class_0
	corpus_1["category"] = class_1

	# Creation du dataset final en regroupant les documents des deux classes
	merged_corpus = pd.concat([corpus_1, corpus_0]) 

	# Recuperation du lemmatizer
	stopwords = nltk.corpus.stopwords.words(language)
	print("os.getcwd() =", os.getcwd())
	# mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))
	if(language == "french"):
		lemmatizer = FrenchLefffLemmatizer()
	elif(language == "english"):
		lemmatizer = WordNetLemmatizer() #le lemmatizer WordNetLemmatizer de nltk uniquement pour l'anglais 

	# Execution de la fonction principale qui fait le nettoyage
	merged_corpus["message_preprocessed"] = preprocess_list_of_documents(merged_corpus['message'], lemmatizer, stopwords)

	# Creation de l'id unique
	merged_corpus.index = list(range(len(merged_corpus)))
	merged_corpus["id"] = merged_corpus.index

	# Suppression des colonnes inutiles
	merged_corpus = merged_corpus[["id", "message", "message_preprocessed", "category"]]

	# Creation de la taille de chaque documents (en nombre de caracteres)
	merged_corpus["length"] = merged_corpus["message"].str.len()

	# Annotation au format entier (necessaire pour certaines fonctions de sklearn)
	merged_corpus["category_bin"] = np.select([merged_corpus["category"] == class_1], [1], default=0)

	# Melange aleatoire des documents
	merged_corpus = merged_corpus.sample(frac=1).reset_index(drop=True)

	# Suppression des retours a la ligne \n et \r
	merged_corpus.replace("\\n", " ", regex=True, inplace=True)
	merged_corpus.replace("\\r", " ", regex=True, inplace=True)

	# Suppression des doublons
	print("merged_corpus.shape =", merged_corpus.shape)
	merged_corpus.drop_duplicates("message", inplace=True, keep="first")
	print("merged_corpus.shape =", merged_corpus.shape)

	#pour enlever les faux exemples, preprocessing restant =
	#  enlever commentaires en bas d'article, description auteur, texte anglais, references bibliographiques
	#  enlever ponctuations (guillemets par exemple) 

	#Credit source : 
	#https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/

	return(merged_corpus)


def save_merged_corpus_table(corpus, class_0, class_1, table_extension):
	# Enregistrer le corpus (au format parquet)
	if not os.path.exists("./data/input/merged_corpus/"):
		os.makedirs("./data/input/merged_corpus/")
	path = "./data/input/merged_corpus/corpus_" + class_1 + "_" + class_0 + "." + table_extension
	if(table_extension == "csv"):
		corpus.to_csv(path, index=False, encoding="utf-8")
		corpus = pd.read_csv(path)
		print(corpus)
	elif(table_extension == "parquet"):
		corpus.to_parquet(path, engine="fastparquet")
		corpus = pd.read_parquet(path) #engine="fastparquet"
		print(corpus)


def get_merged_corpus_table_from_csv(filename_corpus_csv):
	""" Renvoie le corpus pandas dataframe a partir d'un corpus au format csv"""
	corpus = pd.read_csv("./data/input/merged_corpus/" + filename_corpus_csv)
	return(corpus)


def get_merged_corpus_table_from_parquet(filename_corpus_parquet):
	""" Renvoie le corpus pandas dataframe a partir d'un corpus au format parquet"""
	corpus = pd.read_parquet("./data/input/merged_corpus/" + filename_corpus_parquet)
	return(corpus)


def get_merged_corpus_table_from_filename(filename_corpus):
	""" Renvoie le corpus pandas dataframe a partir d'un corpus au format parquet"""
	file_extension = get_file_extension(filename_corpus)
	if(file_extension == "csv"):
		corpus = get_merged_corpus_table_from_csv(filename_corpus)
	elif(file_extension == "parquet"):
		corpus = get_merged_corpus_table_from_parquet(filename_corpus)
	return(corpus)