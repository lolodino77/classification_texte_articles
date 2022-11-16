import os
import glob
import requests
import re
from bs4 import BeautifulSoup
from lib_general import *
from lib_create_corpus import *

# TO DO : preciser que l'extraction sitemap ne marche qu'avec les blogs blogspot (ne marche pas avec wordpress)

# Liste des fonctions : 


def get_bibliographies_list_from_file(filename):
	""" Donne une liste de bibliographies (urls) a partir d'un fichier texte """
	bibliographies_list = open("./data/input/bibliographies/{}".format(filename))
	bibliographies_list = bibliographies_list.read().split("\n")
	return(bibliographies_list)


def get_articles_from_bibliography(bibliography_url):
	""" Recupere tous les articles presents en lien hypertexte sur une page web bibliographie
	
	Parametres: 
	bibliography_url (liste de string) : L'url de la page web bibliographie dont on veut extraire les articles 
		Ex : https://parlafoi.fr/lire/series/commentaire-de-la-summa/
	
	Sortie:
 	articles_urls (liste de string) : La liste des urls des articles presents sur une page web bibliographie
	"""
	articles_urls = get_all_articles_from_webpage(bibliography_url)
	return(articles_urls)