import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import html2text
import nltk
import re
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.stem import WordNetLemmatizer
from pathlib import Path, PureWindowsPath
pd.set_option('display.max_colwidth', 30)
from lib_create_articles_lists import *
from lib_general import *
# from lib_create_corpus_method_bibliographies import *
 	

# Liste des fonctions :



def get_paragraphs_of_article(article_url):
	"""Ecrit les paragraphes d'un article dans un fichier texte
	
	Parametres: 
	article_url (string) : L'url de l'article a decouper en plusieurs parties
	
	Sortie:
 	None : Fichier output_filename qui contient les documents de l'article dont l'url est article_url
	"""
	#Recupere le texte de la page web a l'aide d'un parser
	
	#ancien
	# page = requests.get(article_url)
	# soup = BeautifulSoup(page.content, 'html.parser')
	# visible_text = soup.get_text() #tout le texte dans une seule string

	# Recupere le texte d'un article mais avec des balises html (<\p><p> par exemple)
	page = requests.get(url=article_url)
	soup = BeautifulSoup(page.content, 'html.parser')
	txt = str(soup) 

	# Conversion des indicateurs de paragraphes et de sections /p et /li en retours a la ligne \n pour le split
	txt = txt.replace("\n", " ")
	txt = txt.replace("</p>", "</p>\n\n")
	txt = txt.replace("<li>", "<p>")
	txt = txt.replace("</li>", "</p>\n\n")
	
	# Suppression des balises html
	txt = html2text.html2text(txt)

	# Decoupage en plusieurs parties avec pour separateur le retour a la ligne \n
	txt = txt.split("\n\n") 
	# print("txt")
	# print(txt)

	#Enleve les paragraphes avec trop peu de caracteres
	txt = [paragraphe for paragraphe in txt if len(paragraphe) > 12] 
	
	#Enleve les paragraphes avec des phrases trop courtes (trop peu de mots)
	txt = [paragraphe for paragraphe in txt if len(paragraphe.split(" ")) > 10]
	
	return(txt)


def save_paragraphs(paragraphs, path_corpus, file_open_mode="w", sep = "\n\n"):
	""" Sauvegarde les paragraphes d'un article dans un fichier texte """
	save_list_to_txt(paragraphs, path_corpus, file_open_mode, sep)


def save_corpus_from_articles_lists(articles_urls, path_corpus, all_articles, num_articles, savemode="overwrite"):
	"""Cree un corpus d'un topic (au format de liste de documents/textes) dans le fichier texte filename_output
	a partir d'une liste d'adresses urls d'articles

	Parametres: 
	articles_urls (liste de string) : La liste d'urls d'articles dont on veut extraire les paragraphes. 
										  Ex : https://parlafoi.fr/lire/series/commentaire-de-la-summa/
	path_articles_list (string) : La liste des paths des listes d'articles
	path_corpus (string) : Le path vers le corpus
	save_mode (string) : Le mode d'ecriture du fichier ("append" = ajouter ou "overwrite" = creer un nouveau)
	
	Sortie:
 	None : Fichier filename_corpus qui contient le corpus, une suite de textes separes par une saut de ligne

	Done : version "overwrite" recreer le corpus a chaque fois de zero 
	To do : version "append" ajouter du texte a un corpus deja cree, version "ignore" ne fais rien si fichier existe deja
			version "error" qui renvoie une erreur si fichier existe deja
	"""
	#Ecrit dans le fichier texte filename_corpus.txt tous les paragraphes tous les articles d'une liste
	if(not all_articles):
		articles_urls = articles_urls[:num_articles] # garder que les num_articles premiers articles
		# rajouter cas ou il n'y a qu'un seul article
	if(savemode == "overwrite"):
		# print("path_corpus =", path_corpus)
		# print("articles_urls =", articles_urls)
		article_url = articles_urls[0]
		print("article_url =")
		print(article_url)
		paragraphs = get_paragraphs_of_article(article_url)
		save_paragraphs(paragraphs, path_corpus, file_open_mode="w", sep = "\n\n")
		for article_url in articles_urls[1:]:
			print("article_url =")
			print(article_url)
			paragraphs = get_paragraphs_of_article(article_url)
			save_paragraphs(paragraphs, path_corpus, file_open_mode="a", sep = "\n\n")
	elif(savemode == "append"):
		for article_url in articles_urls:
			print("article_url =")
			print(article_url)
			paragraphs = get_paragraphs_of_article(article_url)
			save_paragraphs(paragraphs, path_corpus, file_open_mode="a", sep = "\n\n")

	# elif(savemode == "ignore"):
	# elif(savemode == "error"):


def preprocess_list_of_documents(list_of_documents, lemmatizer, stopwords):
	"""Nettoie tous les documents d'une liste pour creer un dataset exploitable par des modeles d'IA.
	
	Parametres:
	list_of_documents (liste de string) : Une liste de documents (les textes a classifier) a nettoyer 
	lemmatizer (fonction) : Le lemmatizer qui servira a lemmatizer les mots des documents si possible
	stopwords (liste de string) : La liste des stopwords (mots frequents mais inutiles a enlever)

	Sortie:
	preprocess_list (liste de string) : Une liste de documents nettoyes
	"""
# cas speciaux restants a traiter :
# mots avec un apostrophe avant (Traite)
# mots composes avec un ou plusieurs tirets (A traiter)
	preprocess_list = []
	for document in list_of_documents :
		#remplacer les virgules bizarres
		document = document.replace("’", "'")

		# supprimer les mots avant les apostrophes (particules comme l', t', etc.)
		document = re.sub(r"\s\w+'", " ", document, 0)

		# enlever la ponctuation et met en minuscule
		ponctuation_to_remove = string.punctuation.replace("-", "")
		document_w_punct = "".join([i.lower() for i in document if i not in ponctuation_to_remove])

		# enlever les chiffres
		document_w_num = ''.join(i for i in document_w_punct if not i.isdigit())

		# transformer les phrases en liste de tokens (en liste de mots)
		tokenize_document = nltk.tokenize.word_tokenize(document_w_num)

		# enlever les stopwords (mots n’apportant pas de sens)
		words_w_stopwords = [i for i in tokenize_document if i not in stopwords]

		# lemmatizer (convertir en la racine)
		words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords) #words_lemmatize est un iterateur
		words_lemmatize = list(words_lemmatize)

		# reformer la phrase en reliant les mots precedents
		document_clean = " ".join(words_lemmatize)

		#rajouter la phrase dans la liste
		preprocess_list.append(document_clean)
		
	return preprocess_list


def get_corpus_table(filename_corpus_txt):
	""" Renvoie le corpus pandas dataframe a partir d'un corpus au format .txt"""
	# print("type filename_corpus_txt =", type(filename_corpus_txt))
	res = open("./data/input/corpus_txt/" + filename_corpus_txt, "r", encoding="utf-8").read().split("\n\n")
	res = [elt for elt in res if len(elt) > 1]
	message = res
	length = [len(elt) for elt in res]
	list_of_rows = list(zip(message, length))
	df = pd.DataFrame(list_of_rows, columns=["message", "length"])

	return(df)


def get_multiple_corpus_table():
	all_corpus_txt = get_all_files_from_a_directory(path_to_directory="./data/input/corpus_txt/")
	return(all_corpus_txt)


# TO DO : creer une fonction generale save_corpus_table ?? pour regrouper save_corpus_table_from_textfile 
# et save_corpus_table_from_dataframe
def save_corpus_table_from_textfile(filename_corpus_txt, corpus_topic, table_extension):
	""" Cree un corpus sous forme de table (csv ou parquet) a partir d'un corpus au format texte .txt """
	print("filename_corpus_txt =", filename_corpus_txt)
	corpus = get_corpus_table(filename_corpus_txt)
	filename_corpus_table = "corpus_{}.{}".format(corpus_topic, table_extension)

	if(table_extension == "csv"):
		if not os.path.exists("./data/input/corpus_{}/".format(table_extension)):
			os.makedirs("./data/input/corpus_{}/".format(table_extension))
		corpus.to_csv("./data/input/corpus_csv/" + filename_corpus_table, index=False, encoding="utf-8")
	elif(table_extension == "parquet"):
		if not os.path.exists("./data/input/corpus_{}/".format(table_extension)):
			os.makedirs("./data/input/corpus_{}/".format(table_extension))
		corpus.to_parquet("./data/input/corpus_parquet/" + filename_corpus_table)


def save_corpus_table_from_dataframe(corpus, corpus_topic, table_extension):
	""" Cree un corpus sous forme de table (csv ou parquet) a partir d'un dataframe pandas """
	filename_corpus_table = "corpus_{}.{}".format(corpus_topic, table_extension)

	if(table_extension == "csv"):
		if not os.path.exists("./data/input/corpus_{}/".format(table_extension)):
			os.makedirs("./data/input/corpus_{}/".format(table_extension))
		corpus.to_csv("./data/input/corpus_csv/" + filename_corpus_table, index=False, encoding="utf-8")
	elif(table_extension == "parquet"):
		if not os.path.exists("./data/input/corpus_{}/".format(table_extension)):
			os.makedirs("./data/input/corpus_{}/".format(table_extension))
		corpus.to_parquet("./data/input/corpus_parquet/" + filename_corpus_table)


def save_multiple_corpus_table_from_textfile(filenames_corpus_txt, corpus_topics, table_extension):
	"""Cree un corpus d'un topic au format pandas dataframe dans le fichier texte filename_output
	
	Parametres: 
	filenames_corpus_txt (liste de string) : La liste des corpus txt a enregistrer au format table (csv ou parquet)
	corpus_topics (liste de string) : Les topics de chaque corpus
	table_extension (string) : L'extension de la table de sortie
					Exemple : output_file_extension = "csv" ou = "parquet"
	
	Sortie:
 	None : Fichier filename_corpus_output qui contient le corpus sous forme de dataframe
	"""
	print("in save_multiple_corpus_table_from_textfile")
	print("corpus texts to save =", filenames_corpus_txt)
	for i in range(len(filenames_corpus_txt)):
		filename_corpus_txt = filenames_corpus_txt[i]
		print("corpus text to save =", filename_corpus_txt)
		corpus_topic = corpus_topics[i]
		print("corpus_topic =", corpus_topic)
		# print("corpus =")
		# print(corpus)
		save_corpus_table_from_textfile(filename_corpus_txt, corpus_topic, table_extension)