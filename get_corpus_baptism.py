from lib_scraping import *

filename = "urls_baptism_articles.txt"

#Ecrit dans le fichier urls_baptism_articles.txt tous les liens url presents dans ces articles
url = "https://parlafoi.fr/lire/series/le-pedobapteme/"
urls = get_urls_on_webpage(url, filename, "w")

#Ecrit dans le fichier corpus_baptism.txt tous les paragraphes de philosophie 
#de chaque article du corpus
output_filename = "corpus_baptism.txt"
file_open_mode = "a"
for url in urls:
	print("url =")
	print(url)
	get_parts_of_article(url, output_filename, file_open_mode)