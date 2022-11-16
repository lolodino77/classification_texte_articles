from lib_general import *
from lib_create_articles_lists import *
from lib_create_articles_lists_method_bibliographies import *
from lib_create_corpus import *


def get_var_num_articles(sys_argv):
	""" Recupere la variable num_articles depuis la commande de terminal """
	num_articles = sys_argv[1]
	if(num_articles != "all"):
		num_articles = int(num_articles) 

	return(num_articles)


def get_var_all_articles(sys_argv):
	""" Recupere la variable all_articles depuis la commande de terminal """
	num_articles = sys_argv[1]
	if(num_articles == "all"):
		all_articles = True
	else:
		all_articles = False
	return(all_articles)


def get_var_table_extension(sys_argv):
	""" Recupere la variable table_extension depuis la commande de terminal 
	python 0_create_corpus_from_bibliographies.py all csv """
	table_extension = sys_argv[2]

	return(table_extension)


def get_bibliographies_filenames(sys_argv):
	"""Obtenir le nom des fichiers de datasets pour l'execution 
	   du script 0_create_corpus_from_bibliographies.py

	Parametres: 
	sys_argv (liste de string) : Les arguments de commande pour executer 2_model_selection.py
		Exemples de commandes de terminal qui lancent un script appelant get_input_filenames : 
		python 0_create_corpus_from_bibliographies.py all csv ./data/input/bibliographies/ txt
		python 0_create_corpus_from_bibliographies.py 8 csv ./data/input/bibliographies/ txt
		python 0_create_corpus_from_bibliographies.py all csv command parquet bibliography_middle_age_fr.txt
		python 0_create_corpus_from_bibliographies.py 8 csv command parquet bibliography_middle_age_fr.txt
		python 0_create_corpus_from_bibliographies.py all csv command parquet bibliography_middle_age_fr.txt bibliography_baptism_fr.txt
		python 0_create_corpus_from_bibliographies.py 8 csv command parquet bibliography_middle_age_fr.txt bibliography_baptism_fr.txt
	Sortie:
	filenames (liste de string) : Le nom des fichiers de datasets pour l'execution du script 0_create_corpus_from_bibliographies.py
		Cas 1 filenames = "command" : les fichiers sont ceux entres dans la commande
		Cas 2 filenames = un path : les fichiers sont tous ceux d'une meme extension dans un dossier
	"""
	#argv[0] = le nom du fichier python execute
	files_to_open = sys_argv[3] # argument du script, si files_to_open==command execute le script sur les 
	# fichiers (datasets) entres en arguments dans la ligne de commande, 
	# mais si files_to_open!=command execute le script sur tous les fichiers du dossier ./data/input

	files_format = sys_argv[4] # format des fichiers datasets a ouvrir (parquet, csv, etc.), multiple si plusieurs formats
	print("len(sys_argv) =", len(sys_argv))
	# sert quand files_to_open==in_input_repertory, pour n'importer que les parquet, ou que les csv, etc.

	if(files_to_open == "command"):
		if(len(sys_argv) == 6): # cas quand il n'y a qu'un seul dataset => il faut creer une liste
			filenames = [sys_argv[5]]
		else: #cas quand il y a au moins deux datasets => pas besoin de creer de liste
			filenames = sys_argv[5:] # ignorer les 4 premiers arguments, nom du script, num_artciles, files_to_open et table_extension
	else:
		input_repertory = files_to_open.replace("/", "\\") # "/data/input/" ==> '\\data\\input\\'
		print("files in input : input_repertory =", input_repertory)
		filenames = glob.glob(os.path.join(input_repertory + "*." + files_format))
		filenames = [filename.split(input_repertory)[1] for filename in filenames] # enlever le path entier, garder que le nom du fichier

	return(filenames)


# TO DO : rajouter et jouer avec l'option file_open_mode pour ajouter a un fichier deja existant
def save_one_corpus_from_bibliographies_lists(bibliography_urls, filename_corpus,
											 all_articles=True, num_articles=0, savemode="overwrite"):
	"""Cree un corpus d'un topic (au format de liste de documents/textes) dans le fichier texte filename_output
	a partir d'une liste d'adresses urls de bibliographies d'articles

	Parametres: 
	bibliography_urls (liste de string) : La liste des urls de bibliographies d'articles dont on veut recuperer
										  les urls. 
										  Ex : [https://parlafoi.fr/lire/series/commentaire-de-la-summa/, ...]
	filename_corpus (string) : Le nom du fichier dans lequel on ecrira le corpus
							   Exemple : corpus_philosophy.txt
	file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
	
	Sortie:
	filename_corpus (string) : Le nom filename_corpus qui contient le corpus, une suite de textes separes par une saut de ligne
	"""
	topic = get_topic_from_filename(filename_corpus, keep_language=True)
	path_articles_list = "./data/input/articles_lists/articles_list_{}.txt".format(topic)
	path_corpus = "./data/input/corpus_txt/" + filename_corpus

	print("in save_one_corpus_from_bibliographies_lists\n")
	# print("filename_corpus =", filename_corpus)
	# print("topic =", topic)
	# print("bibliography_urls =", bibliography_urls)

	if(savemode == "overwrite"):
		bibliography_url = bibliography_urls[0]
		print("bibliography_url(0) =", bibliography_url)
		articles_urls = get_articles_from_bibliography(bibliography_url)
		# print("articles_urls = get_articles_from_bibliography(bibliography_url)")
		if(not all_articles):
			print("articles_urls =", articles_urls[:num_articles])
		save_corpus_from_articles_lists(articles_urls, path_corpus, all_articles, num_articles, savemode="overwrite")
		for bibliography_url in bibliography_urls[1:]:
			print("\nbibliography_url =", bibliography_url)
			# Recupere tous les articles d'une bibliographie
			articles_urls = get_articles_from_bibliography(bibliography_url)
			print("articles_urls =", articles_urls[:2])
			save_articles_lists(articles_urls, path_articles_list, file_open_mode="w", sep = "\n")

			# Cree un corpus a partir d'une liste d'urls d'articles
			save_corpus_from_articles_lists(articles_urls, path_corpus, all_articles, num_articles, savemode="append")
	elif(savemode == "append"):
		for bibliography_url in bibliography_urls:
			# Recupere tous les articles d'une bibliographie
			articles_urls = get_articles_from_bibliography(bibliography_url)
			# print("articles_urls (get_articles_from_bibliography) =", articles_urls)
			save_articles_lists(articles_urls, path_articles_list, file_open_mode="w", sep = "\n")

			# Cree un corpus a partir d'une liste d'urls d'articles
			save_corpus_from_articles_lists(articles_urls, path_corpus, all_articles, num_articles, savemode="append")


def save_multiple_corpus_from_bibliographies_lists_files(bibliographies_filenames, all_articles=True,
															num_articles=0):
	"""Cree un corpus au format texte a partir de fichiers listes de bibliographies (urls) 
		= (pages web qui listent des articles) 
	TO DO : rajouter une version savemode="append" pour elargir un corpus deja existant
	"""
	filenames_corpus_txt = []
	corpus_topics = []
	for filename in bibliographies_filenames:
		# Lecture des fichiers "listes de bibliographies"
		print("filename =", filename)
		bibliographies_list = get_bibliographies_list_from_file(filename)
		print("----------------------------------------------------------------")
		print("----------------------------------------------------------------")
		print("\n\nbibliographies_list =", bibliographies_list)
		corpus_topic = get_topic_from_filename(filename, keep_language=True)
		filename_corpus_txt = "corpus_{}.txt".format(corpus_topic)

		# Ecrire le corpus a partir des bibliographies
		save_one_corpus_from_bibliographies_lists(bibliographies_list, filename_corpus_txt,
												all_articles, num_articles, savemode="overwrite")
		filenames_corpus_txt.append(filename_corpus_txt)
		corpus_topics.append(corpus_topic)

	return(filenames_corpus_txt, corpus_topics)

