import os
import glob
import requests
import re
from bs4 import BeautifulSoup
from lib_general import *

# Liste des fonctions
# makedir_articles_lists
# get_topic_from_filename(filename, keep_language)
# save_articles_lists(articles_urls, path_articles_list, file_open_mode="w", sep = "\n")
# get_all_articles_from_webpage
# save_articles_lists(articles_urls, path_articles_list, file_open_mode="w", sep = "\n"):
# save_list_to_txt(articles_urls, path_articles_list, file_open_mode, sep)
# get_all_articles_from_webpage(url)


def makedir_articles_lists():
	""" Cree le dossier articles_lists s'il n'existe pas deja """
	if not os.path.exists("./data/input/articles_lists/"):
		os.makedirs('./data/input/articles_lists/')


def get_topic_from_filename(filename, keep_language):
	"""Extrait le topic qui apparait dans le nom d'un fichier.
	
	Parametres:
	filename (string) : Le nom du fichier duquel on veut extraire le topic
						Au format : corpus_ + topic + langue + extension
						Exemple : "corpus_philosophy_fr.txt", "corpus_animals.csv"
	keep_language (boolean) : Indique s'il garder la langue dans le topic
						Exemples : si keep_language==True ==> philosophy_fr
								   sinon ==> philosophy

	Sortie:
	topic (string) : Le topic (sujet/theme) extrait du nom de fichier filename
					 Exemple : "philosophy_fr", "animals"
	"""
	# version sans langue ("fr", "eng") dans filename
	print("filename avant split =", filename)
	topic = filename.split(".")[0].split("_")[1:]
	topic = "_".join(topic)

	# topic = filename.split(".")[0].split("corpus_")[1] # corpus_beaux_arts_fr.txt => beaux_arts 

	# version avec langue ("fr", "eng") dans filename
	# filename = filename.split(".")[0]
	# topic = filename.split("_")[1:][0] 
	# if(keep_language == True):
	# 	topic = "_".join(topic)
	# else:
	# 	topic = "_".join(topic[:-1])
	
	return(topic)


def save_articles_lists(articles_urls, path_articles_list, file_open_mode="w", sep = "\n"):
	""" # Enregistrer la liste des urls d'articles """
	save_list_to_txt(articles_urls, path_articles_list, file_open_mode, sep)


def get_all_articles_from_webpage(url):
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
		link_str = str(link)
		# print("link =", str(link))
		# print("type(link) =", type(link))
		if("https" in link_str):
			url_i = link.get('href')
			if("/20" in url_i): # condition si c'est un article (20 = 2 premiers chiffres des annees 2021, 2010...)
				urls.append(url_i)
	urls = [url for url in urls if("pdf" not in url)] # enlever les articles pdf

	return(urls)


# def write_articles_list_from_webpage(url, filename, file_open_mode):
#     """Ecrit dans un fichier texte la liste des urls d'articles (liens hypertextes) 
#     presents sur une page internet.

#     Parametres:
#     url (string) : L'url de la page internet dont on veut recuperer les urls (une bibliographie)
#     filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls
#     file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
#     """
#     #Ecrit le resultat dans un fichier texte
#     articles_urls = get_all_articles_from_webpage(url)
#     path_to_file = "./data/input/articles_lists/" + filename
#     save_list_to_txt(articles_urls, path_to_file, file_open_mode, sep = "\n")


#sert a rien
# def write_articles_list_from_multiple_webpages(file_list_bibliographies, new_file=True):
#     """Ecrit dans un fichier texte la liste des urls d'articles (liens hypertextes) 
#     presents sur une page internet.

#     Parametres:
#     file_list_bibliographies (string) : Les urls des pages internet dont on veut recuperer les urls 
#                                         (les bibliographies)
#     articles_list_filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls
#     new_file (string) : Precise s'il faut creer un nouveau fichier de sortie articles_list_filename ou non
#     """
#     urls = open("./data/input/bibliographies/{}".format(file_list_bibliographies))
#     urls = urls.read().split("\n")
#     print("urls =", urls)

#     if(new_file):
#         file_open_mode = "w"
#     else:
#         file_open_mode = "a"

#     url = urls[0]
#     print("url =", url)
#     topic = get_topic_from_filename(file_list_bibliographies, keep_language=True)
#     articles_list_filename = "articles_list_{}.txt".format(topic)
#     write_articles_list_from_webpage(url, articles_list_filename, file_open_mode)

#     if(len(urls) > 1):
#         for url in urls[1:]: #ignorer le premier url deja traite
#             print("url =", url)
#             write_articles_list_from_webpage(url, articles_list_filename, "a")