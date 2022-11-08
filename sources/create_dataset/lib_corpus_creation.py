import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import html2text
 	
# Liste des fonctions :
# get_urls_on_webpage(url, filename, file_open_mode)
# write_paragraphs_of_article(article_url, output_filename, file_open_mode)
# write_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode)
# write_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode)


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


def write_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode="a"):
	"""Cree un corpus (une liste de documents/textes) dans le fichier texte filename_output
	
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
	print("in write_corpus_from_urls")
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


def write_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode):
	"""Cree un corpus (une liste de documents/textes) dans le fichier texte filename_output
	
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
