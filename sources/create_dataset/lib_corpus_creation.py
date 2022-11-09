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

 	
# Liste des fonctions :
# get_urls_on_webpage(url, filename, file_open_mode)
# write_paragraphs_of_article(article_url, output_filename, file_open_mode)
# write_topic_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode)
# write_topic_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode)


def get_topic_from_filename(filename, keep_language):
	"""Extrait le topic qui apparait dans le nom d'un fichier.
	
	Parametres:
	filename (string) : Le nom du fichier duquel on veut extraire le topic
						Au format : structure_de_donnees + topic + langue + extension
						Exemple : "dataset_philosophy_fr.txt", "corpus_animals.csv"
	keep_language (boolean) : Indique s'il garder la langue dans le topic
						Exemples : si keep_language==True ==> philosophy_fr
								   sinon ==> philosophy

	Sortie:
	topic (string) : Le topic (sujet/theme) extrait du nom de fichier filename
					 Exemple : "philosophy_fr", "animals"
	"""
	filename = filename.split(".")[0]
	topic = filename.split("_")[1:] 
	if(keep_language == True):
		topic = "_".join(topic)
	else:
		topic = "_".join(topic[:-1])
	
	return(topic)


def get_urls_on_webpage(url, filename, file_open_mode):
	"""Ecrit dans un fichier texte la liste des urls (liens hypertextes) presents 
	sur une page internet.
	
	Parametres:
	url (string) : L'url de la page internet dont on veut recuperer les urls
	filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls sur url
	file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
	
	Sortie:
	urls (liste de string) : Une liste d'urls + (ecriture du resultat dans le fichier filename)
	"""
	#Recupere le texte de la page web a l'aide d'un parser
	reqs = requests.get(url)
	soup = BeautifulSoup(reqs.text, 'html.parser')
	
	#Recupere un par un tous les liens url presents sur l'article
	urls = []
	for link in soup.find_all('a'):
		url_i = link.get('href')
		if(url_i[0:22] == "https://parlafoi.fr/20"):
			urls.append(url_i)

	# Se placer dans le bon dossier pour ecrire le resultat
	os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..' * 2)))

	#Ecrit le resultat dans un fichier texte
	f = open("./data/input/" + filename, file_open_mode)
	for url in urls:
		f.write(url + "\n")
	f.close()

	return(urls)


def write_paragraphs_of_article(article_url, output_filename, file_open_mode):
	"""Ecrit les paragraphes d'un article dans un fichier texte
	
	Parametres: 
	article_url (string) : L'url de l'article a decouper en plusieurs parties
	output_filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls sur url
	file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
	
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

	#Ecrit le resultat dans un fichier texte
	f = open(output_filename, file_open_mode, encoding="utf-8")
	for paragraphe in txt:
		f.write(paragraphe + "\n\n") #saut de ligne pour pouvoir distinguer les paragraphes
	f.close()


def write_topic_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode="a"):
	"""Cree un corpus d'un topic (au format de liste de documents/textes) dans le fichier texte filename_output
	
	Parametres: 
	filename_urls_articles (string) : Le nom du fichier dans lequel on ecrira la liste des urls des articles
	filename_corpus (string) : Le nom du fichier dans lequel on ecrira le corpus
							   Exemple : corpus_philosophy.txt
	bibliography_urls (liste de string) : La liste des urls de bibliographies d'articles dont on veut recuperer
										  les urls. 
										  Ex : https://parlafoi.fr/lire/series/commentaire-de-la-summa/
	file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
	
	Sortie:
 	None : Fichier filename_corpus qui contient le corpus, une suite de textes separes par une saut de ligne
	"""
	# Se placer dans le bon dossier pour ecrire le resultat
	os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..' * 2)))
	print("in write_topic_corpus_from_urls")
	print("bibliography_urls =", bibliography_urls)
	print("type of bibliography_urls =", type(bibliography_urls))
	for bibliography_url in bibliography_urls:
		print("bibliography_url =", bibliography_url)
		articles_urls = get_urls_on_webpage(bibliography_url, filename_urls_articles, "a")	

		#Ecrit dans le fichier texte corpus_philosophy.txt tous les paragraphes de philosophie
		#de chaque article du corpus 
		file_open_mode = "a"
		for article_url in articles_urls:
			print("article_url =")
			print(article_url)
			write_paragraphs_of_article(article_url, "./data/input/" + filename_corpus, file_open_mode)


def write_topic_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode):
	"""Cree un corpus d'un topic au format pandas dataframe dans le fichier texte filename_output
	
	Parametres: 
	filename_corpus_input (string) : Le nom du fichier dans lequel se trouve le corpus en suite de textes
							   Exemple : corpus_philosophy.txt
	filename_corpus_output (string) : Le nom du fichier dans lequel on ecrira le corpus sous format csv
							   Exemple : dataset_philosophy.txt
	file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
	
	Sortie:
 	None : Fichier filename_corpus_output qui contient le  corpus sous forme de dataframe
	"""
	# Se placer dans le bon dossier pour lire les entrees et ecrire le resultat
	os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..' * 2)))

	# print("tonpere")
	# res = open("./data/input/" + filename_corpus_input, "r").read().split("\n\n")
	res = open("./data/input/" + filename_corpus_input, "r", encoding="utf-8").read().split("\n\n")
	print(res)
	res = [elt for elt in res if len(elt) > 1]

	message = res
	length = [len(elt) for elt in res]
	list_of_rows = list(zip(message, length))

	df = pd.DataFrame(list_of_rows, columns=["message", "length"])
	print(df.head(20))
	print(df.shape)

	df.to_csv("./data/input/" + filename_corpus_output, index=False, encoding="utf-8")


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


def write_multiple_topics_corpus_dataset(corpus_datasets_names, final_corpus_name, topics, language):
	"""Cree un corpus d'un topic au format pandas dataframe dans le fichier texte filename_output
	Marche pour l'instant que pour fusionner deux corpus (deux topics differents)
	To do : faire pour classification multiclasse

	Parametres: 
	corpus_datasets_names (liste de string) : Les noms des datasets de corpus a fusionner
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
	corpus_0 = pd.read_csv('data/input/' + corpus_datasets_names[0])
	corpus_1 = pd.read_csv('data/input/' + corpus_datasets_names[1])
	
	# Annotation des documents
	class_0 = topics[0]
	class_1 = topics[1]
	corpus_0["category"] = class_0
	corpus_1["category"] = class_1

	# Creation du dataset final en regroupant les documents des deux classes
	corpus = pd.concat([corpus_1, corpus_0]) 

	# Recuperation du lemmatizer
	stopwords = nltk.corpus.stopwords.words(language)
	print("os.getcwd() =", os.getcwd())
	# mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))
	if(language == "french"):
		lemmatizer = FrenchLefffLemmatizer()
	elif(language == "english"):
		lemmatizer = WordNetLemmatizer() #le lemmatizer WordNetLemmatizer de nltk uniquement pour l'anglais 

	# Execution de la fonction principale qui fait le nettoyage
	corpus["message_preprocessed"] = preprocess_list_of_documents(corpus['message'], lemmatizer, stopwords)

	# Creation de l'id unique
	corpus.index = list(range(len(corpus)))
	corpus["id"] = corpus.index

	# Suppression des colonnes inutiles
	corpus = corpus[["id", "message", "message_preprocessed", "category"]]

	# Creation de la taille de chaque documents (en nombre de caracteres)
	corpus["length"] = corpus["message"].str.len()

	# Annotation au format entier (necessaire pour certaines fonctions de sklearn)
	corpus["category_bin"] = np.select([corpus["category"] == class_1], [1], default=0)

	# Melange aleatoire des documents
	corpus = corpus.sample(frac=1).reset_index(drop=True)

	# Suppression des retours a la ligne \n et \r
	corpus.replace("\\n", " ", regex=True, inplace=True)
	corpus.replace("\\r", " ", regex=True, inplace=True)

	# Suppression des doublons
	print("corpus.shape =", corpus.shape)
	corpus.drop_duplicates("message", inplace=True, keep="first")
	print("corpus.shape =", corpus.shape)

	#pour enlever les faux exemples, preprocessing restant =
	#  enlever commentaires, description auteur, texte anglais, references bibliographiques
	#  enlever ponctuations (guillemets par exemple) 

	# Enregistrer le corpus
	print("finnnnnnn")
	path = "./data/input/data_" + class_1 + "_" + class_0 + ".parquet"
	corpus.to_parquet(path, engine="fastparquet")
	corpus = pd.read_parquet(path) #engine="fastparquet"
	print(corpus)


def create_individual_topic_corpus(bibliography_filename):
	"""Cree un corpus d'un topic au format pandas dataframe dans un fichier (parquet, csv, etc.) 

	Parametres: 
	bibliography_filename (liste de string) : Les nom du fichier qui contient la bibliographie d'articles pour creer le corpus dataset 
		Exemple : "bibliography_philosophy_fr.txt"

	Sortie:
 	None : Fichier filename_corpus_output qui contient le corpus au format pandas dataframe
	"""
	f = open(os.getcwd() + "\\data\\input\\bibliographies\\" + bibliography_filename, "r")
	bibliography_urls = f.read().split("\n")
	topic = get_topic_from_filename(bibliography_filename, keep_language=True)
	filename_urls_articles = "urls_{}_articles.txt".format(topic)
	filename_corpus = "corpus_{}.txt".format(topic)
	filename_corpus_input = "corpus_{}.txt".format(topic)
	filename_corpus_output = "dataset_{}.csv".format(topic)

	write_topic_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode="a")
	write_topic_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode="a")