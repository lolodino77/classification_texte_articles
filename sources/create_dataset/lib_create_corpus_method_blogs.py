from lib_general import *
from lib_create_articles_lists import *
from lib_create_articles_lists_method_blogs import *
from lib_create_corpus import *


def get_var_num_articles(sys_argv):
	""" Recupere la variable num_articles depuis la commande de terminal """
	num_articles = sys_argv[4]
	if(num_articles != "all"):
		num_articles = int(num_articles) 

	return(num_articles)


def get_var_all_articles(sys_argv):
	""" Recupere la variable all_articles depuis la commande de terminal """
	num_articles = sys_argv[4]
	if(num_articles == "all"):
		all_articles = True
	else:
		all_articles = False
	return(all_articles)


def create_corpus_from_blogs_names(file_list_of_blogs, all_articles, num_articles):
	"""Cree un corpus d'un topic au format pandas dataframe dans un fichier (parquet, csv, etc.) 
	a partir du nom du blog blog_name

	Parametres: 
	file_list_of_blogs (string) : Le nom du fichier qui contient la liste des blogs
		Exemple : "blogs_philosophy_eng.txt"
	num_articles (int) : Le nombre d'articles a garder, vaut 0 si on les garde tous

	Sortie:
 	None : Fichier filename_corpus_output qui contient le corpus au format pandas dataframe
	"""
	blogs_names = get_blogs_from_file(file_list_of_blogs)
	for blog_name in blogs_names:
		print("blog_name =", blog_name)
		author = get_author_from_blog_name(blog_name)
		path_corpus = "./data/input/corpus_txt/corpus_{}.txt".format(author)
		articles_urls = get_all_articles_from_blog(blog_name, all_articles, num_articles=num_articles)
		
		# if(all_articles):
		# 	num_articles = len(articles_urls)
		# articles_urls = articles_urls[:num_articles]

		path_articles_list = "./data/input/articles_lists/articles_list_{}.txt".format(author)
		save_list_to_txt(articles_urls, path_articles_list, file_open_mode="w", sep="\n")
		
		# print("articles_urls =", articles_urls)
		article_url = articles_urls[0]
		print("article_url =", article_url)

		paragraphs = get_paragraphs_of_article(article_url)
		paragraphs = [paragraph for paragraph in paragraphs if("http" not in paragraph)] # enlever les textes qui contiennent "https"
		# print("paragraphs =", paragraphs)
		save_list_to_txt(paragraphs, path_corpus, file_open_mode="w", sep = "\n\n") #"w" cree nouveau fichier
		for article_url in articles_urls[1:]:
			# print("\n\n\n\n\n")
			print("article_url =", article_url)
			paragraphs = get_paragraphs_of_article(article_url)
			paragraphs = [paragraph for paragraph in paragraphs if("http" not in paragraph)] # enlever les textes qui contiennent "https"
			# print("paragraphs =", paragraphs)
			save_list_to_txt(paragraphs, path_corpus, file_open_mode="a", sep = "\n\n") #ecrit en mode append "a"

	
def create_corpus_csv_from_blogs_names(file_list_of_blogs, input_file_extension, output_file_extension,
										all_articles, num_articles):
	"""Cree un corpus csv d'un topic dans un fichier texte
	
	Parametres: 
	file_list_of_blogs (string) : Le nom du fichier qui contient la liste des blogs
					Exemple : "blogs_philosophy_eng.txt"
	input_file_extension (string) : L'extension du fichier dans lequel se trouve le corpus en suite de textes
					Exemple : input_file_extension = "txt"
	output_file_extension (string) : L'extension du fichier dans lequel on ecrira le corpus sous format csv
					Exemple : output_file_extension = "csv"
	"""
	create_corpus_from_blogs_names(file_list_of_blogs, all_articles, num_articles)
	# write_all_corpus_txt_to_corpus_csv(file_list_of_blogs, input_file_extension, output_file_extension)