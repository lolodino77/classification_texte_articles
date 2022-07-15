import requests
from bs4 import BeautifulSoup
import html2text
 	

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

	#Ecrit le resultat dans un fichier texte
	f = open(filename, file_open_mode)
	for url in urls:
		f.write(url + "\n")
	f.close()

	return(urls)


def get_parts_of_article(url, output_filename, file_open_mode):
	"""Decoupe differentes parties d'un article et les ecrit/rajoute dans un fichier texte
	
	Parametres: 
	url (string) : L'url de l'article a decouper en plusieurs parties
	output_filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls sur url
	file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
	
	Sortie:
 	None : ecriture de la liste des parties de l'article dans output_filename
	"""
	#Recupere le texte de la page web a l'aide d'un parser
	
	#ancien
	# page = requests.get(url)
	# soup = BeautifulSoup(page.content, 'html.parser')
	# visible_text = soup.get_text() #tout le texte dans une seule string

	#nouveau
	page = requests.get(url=url)
	soup = BeautifulSoup(page.content, 'html.parser')
	txt = str(soup)
	txt = txt.replace("\n", " ")
	txt = txt.replace("</p>", "</p>\n\n")
	txt = txt.replace("<li>", "<p>")
	txt = txt.replace("</li>", "</p>\n\n")
	txt = html2text.html2text(txt)
	# print("txt")
	# print(txt)

	txt = txt.split("\n\n") #decoupage en plusieurs parties avec separateur saut de ligne
	# print("txt")
	# print(txt)

	#Enleve les parties avec trop peu de caracteres
	txt = [paragraphe for paragraphe in txt if len(paragraphe) > 12] 
	
	#Enleve les parties avec des phrases trop courtes (trop peu de mots)
	txt = [paragraphe for paragraphe in txt if len(paragraphe.split(" ")) > 10]

	#Ecrit le resultat dans un fichier texte
	f = open(output_filename, file_open_mode)
	for paragraphe in txt:
		f.write(paragraphe + "\n\n") #saut de ligne pour pouvoir distinguer les paragraphes
	f.close()