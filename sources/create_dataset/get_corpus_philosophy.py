from lib_scraping import *

filename = "urls_philosophy_articles.txt"

#Ecrit dans urls_philosophy_articles.txt tous les liens url presents dans ces articles
url = "https://parlafoi.fr/lire/series/commentaire-de-la-summa/"
urls = get_urls_on_webpage(url, filename, "w")
url = "https://parlafoi.fr/lire/series/notions-de-base-en-philosophie/"
urls = urls + get_urls_on_webpage(url, filename, "a")
url = "https://parlafoi.fr/lire/series/la-scolastique-protestante/"
urls = urls + get_urls_on_webpage(url, filename, "a")
url = "https://parlafoi.fr/lire/series/le-presuppositionnalisme/"
urls = urls + get_urls_on_webpage(url, filename, "a")
urls = list(set(urls)) #enlever les doublons

#Ecrit dans le fichier texte corpus_philosophy.txt tous les paragraphes de philosophie
#de chaque article du corpus 
output_filename = "corpus_philosophy.txt"
file_open_mode = "a"
for url in urls:
	print("url =")
	print(url)
	get_parts_of_article(url, output_filename, file_open_mode)