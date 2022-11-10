import os
import glob
import requests
import re
from bs4 import BeautifulSoup


def makedir_articles_lists():
    """ Cree le dossier articles_lists s'il n'existe pas deja """
    if not os.path.exists("./data/input/articles_lists/"):
        os.makedirs('./data/input/articles_lists/')


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


def write_list_to_txt(input_list, path_to_file, file_open_mode, sep):
    """ Ecrit une liste input_list (avec saut a la ligne) dans un fichier texte situe au path path_to_file
    Entrees:
        input_list (liste de string) : La liste de string a ecrire dans le fichier texte
        sep (string) : Le separateur entre deux textes du fichier texte (\n, \n\n, etc.)
    """
    f = open(path_to_file, file_open_mode, encoding="utf-8") #"w" si n'existe pas, "a" si on veut ajouter a un fichier deja existant
    for line in input_list:
        f.write(line + sep)
    f.close()
    

def get_web_page_text_contents(url):
    """ Donne dans une string le contenu texte d'une page web simple (avec que du textec comme un fichier texte) """
    page = requests.get(url) #page.text donne le contenu texte d'un page web (comme si c'etait un fichier txt)    
    text_contents = page.text

    return(text_contents)


def get_blog_robots_page(blog_name):
    """ Recupere la page robots.txt d'un blog 
        Exemple : blog_name = "http://alexanderpruss.blogspot.com"
    """
    return(blog_name + "/robots.txt")


def get_sitemap_page(blog_name):
    """ Recupere la page sitemap d'un blog """
    robots_txt_page = get_blog_robots_page(blog_name)
    text_contents = get_web_page_text_contents(robots_txt_page)
    text_contents = text_contents.split("\n")
    sitemap_contents = [elt for elt in text_contents if "Sitemap" in elt]
    print("sitemap_contents =", sitemap_contents)
    sitemap_page = sitemap_contents[0].split(" ")[1]

    return(sitemap_page)


def get_sitemap_from_main_sitemap(sitemap_page):
    """ Donne la liste des urls sitemap presents sur la principale page sitemap d'un blog """
    urls = get_web_page_text_contents(sitemap_page)
    urls = urls.replace('<?xml version=\'1.0\' encoding=\'UTF-8\'?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap>', "")
    urls = urls.replace("</loc></sitemap><sitemap>", "\n")
    urls = urls.replace("</loc></sitemap></sitemapindex>", "")
    urls = urls.replace("<loc>", "")
    urls = urls.split("\n")
    return(urls)


def get_urls_from_one_sitemap_subpage(sitemap_subpage):
    """ Recupere les urls sur une seule sous-page sitemap """
    urls = get_web_page_text_contents(sitemap_subpage)
    urls = urls.replace("""<?xml version='1.0' encoding='UTF-8'?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">""", "")
    urls = urls.replace("</lastmod></url><url><loc>", "")
    urls = urls.replace("</loc><lastmod>", "\n")
    urls = re.sub(r"\d\d\d\d-\d\d-\d\d\w\d\d:\d\d:\d\d\w", "", urls, 0) #pour enlever les string comme 2022-10-21T00:02:02Z
    urls = urls.replace("<url><loc>", "\n")
    urls = urls.replace("\n</lastmod></url></urlset>", "\n")
    urls = urls.split("\n")[1:] #enlever premier element egal a ""
    
    return(urls)


def get_urls_from_all_sitemap_subpages(sitemap_subpages):
    """ Recupere les urls de toutes les sous-pages sitemap """
    urls = []
    for sitemap_subpage in sitemap_subpages:
        urls += get_urls_from_one_sitemap_subpage(sitemap_subpage)
    
    return(urls)


def get_author_from_blog_name(blog_name):
    """ Recupere le nom de l'auteur d'un blog a partir de la page d'accueil de ce blog 
        Exemple : blog_name = "https://edwardfeser.blogspot.com 
    """
    blog_name = blog_name.split("//")[1] # garder la partie apres https:
    author_name = blog_name.split(".")[0] # garder la premiere partie, ce qu'il y a avant blogspot/wordpress etc.
    
    return(author_name)
    

def get_all_articles_from_blog(blog_name):
    """ Renvoie dans une liste tous les articles d'un blog (wordpress ou blogspot) a partir de sa page d'accueil
        Exemple : "https://majestyofreason.wordpress.com/", "https://edwardfeser.blogspot.com"
        urls = ["www.aaa.wordpress.fr\n", "www.aba.wordpress.fr\n", "www.aaa.wordpress.fr"] 
    """
    # Recupere dans une liste urls les adresses url de tous les articles publies d'un blog
    sitemap_page = get_sitemap_page(blog_name)
    urls_sitemap = get_sitemap_from_main_sitemap(sitemap_page)
    urls = get_urls_from_all_sitemap_subpages(urls_sitemap)
    print("check ''", "\n" in urls)

    return(urls)


def write_articles_list_from_blog(blog_name):
    """ Renvoie tous les articles d'un blog (wordpress ou blogspot) a partir de sa page d'accueil
        Exemple : 
            blog_name = "https://majestyofreason.wordpress.com/", "https://edwardfeser.blogspot.com" 
    """
    urls = get_all_articles_from_blog(blog_name)
    author = get_author_from_blog_name(blog_name)
    path_to_articles_list = "./data/input/articles_lists/articles_list_{}.txt".format(author)
    write_list_to_txt(urls, path_to_articles_list, file_open_mode = "w", sep = "\n") # "w" car copie globale du blog entier


def write_articles_lists_from_multiple_blogs(file_list_of_blogs):
    """ Ecrit dans des fichiers textes les urls de tous les articles de plusieurs blogs respectifs listes 
        dans le fichier file_list_of_blogs (wordpress ou blogspot) a partir de leur page d'accueil
        Exemple : file_list_of_blogs = "blogs_philosophy_eng.txt", "blogs_history_fr.txt" 
    """
    blogs = open("./data/input/blogs/{}".format(file_list_of_blogs))
    blogs = blogs.read().split("\n")
    for blog in blogs:
        print("blog =", blog)
        write_articles_list_from_blog(blog)


def get_all_articles_from_webpage(bibliography_url):
	"""Ecrit dans un fichier texte la liste des urls (liens hypertextes) presents 
	sur une page internet.
	
	Parametres:
	bibliography_url (string) : L'url de la page internet dont on veut recuperer les urls
	filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls sur url
	file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
	
	Sortie:
	urls (liste de string) : Une liste d'urls + (ecriture du resultat dans le fichier filename)
	"""
	#Recupere le texte de la page web a l'aide d'un parser
	reqs = requests.get(bibliography_url)
	soup = BeautifulSoup(reqs.text, 'html.parser')
	
	#Recupere un par un tous les liens url presents sur l'article
	urls = []
	for link in soup.find_all('a'):
		url_i = link.get('href')
		# if(url_i[0:22] == "https://parlafoi.fr/20"): # ancienne condition a changer generaliser
		if("/20" in url_i): # condition si c'est un article (20 = 2 premiers chiffres des annees 2021, 2010...)
			urls.append(url_i)

	# Se placer dans le bon dossier pour ecrire le resultat
	# os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..' * 2)))

	return(urls)


def write_articles_list_from_webpage(bibliography_url, filename, file_open_mode):
    """Ecrit dans un fichier texte la liste des urls d'articles (liens hypertextes) 
    presents sur une page internet.

    Parametres:
    bibliography_url (string) : L'url de la page internet dont on veut recuperer les urls (une bibliographie)
    filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls
    file_open_mode (string) : Le mode d'ouverture du fichier ("a", "w", etc.)
    """
    #Ecrit le resultat dans un fichier texte
    articles_urls = get_all_articles_from_webpage(bibliography_url)
    path_to_file = "./data/input/articles_lists/" + filename
    write_list_to_txt(articles_urls, path_to_file, file_open_mode, sep = "\n")


def write_articles_list_from_multiple_webpages(file_list_bibliographies, new_file=True):
    """Ecrit dans un fichier texte la liste des urls d'articles (liens hypertextes) 
    presents sur une page internet.

    Parametres:
    file_list_bibliographies (string) : Les urls des pages internet dont on veut recuperer les urls 
                                        (les bibliographies)
    articles_list_filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls
    new_file (string) : Precise s'il faut creer un nouveau fichier de sortie articles_list_filename ou non
    """
    bibliography_urls = open("./data/input/bibliographies/{}".format(file_list_bibliographies))
    bibliography_urls = bibliography_urls.read().split("\n")
    print("bibliography_urls =", bibliography_urls)

    if(new_file):
        file_open_mode = "w"
    else:
        file_open_mode = "a"

    bibliography_url = bibliography_urls[0]
    print("bibliography_url =", bibliography_url)
    topic = get_topic_from_filename(file_list_bibliographies, keep_language=True)
    articles_list_filename = "articles_list_{}.txt".format(topic)
    write_articles_list_from_webpage(bibliography_url, articles_list_filename, file_open_mode)

    if(len(bibliography_urls) > 1):
        for bibliography_url in bibliography_urls[1:]: #ignorer le premier url deja traite
            print("bibliography_url =", bibliography_url)
            write_articles_list_from_webpage(bibliography_url, articles_list_filename, "a")


def write_multiple_articles_list_from_webpages():
# def write_multiple_articles_list_from_webpages(file_list_bibliographies, articles_list_filename, new_file=True):
    """Ecrit dans des fichiers textes des listes respectives d'urls d'articles (liens hypertextes) 
    presents sur une page internet.

    Parametres:
    file_list_bibliographies (string) : Les urls des pages internet dont on veut recuperer les urls 
                                        (les bibliographies)
    articles_list_filename (string) : Le nom du fichier dans lequel on ecrira la liste des urls
    new_file (string) : Precise s'il faut creer un nouveau fichier de sortie articles_list_filename ou non
    """
    # input_repertory = files_to_open.replace("/", "\\") 
    bibliography_files = glob.glob(os.path.join("./data/input/bibliographies/".replace("/", "\\") + "*.txt"))
    bibliography_files = glob.glob(os.path.join("./data/input/bibliographies/" + "*.txt")) #enlever le path avant les filenames
    bibliography_files = [filename.replace("\\", "/") for filename in bibliography_files]
    bibliography_files = [filename.split("./data/input/bibliographies/")[1] for filename in bibliography_files]
    print("bibliography_files =", bibliography_files)

    for bibliography_file in bibliography_files:
        write_articles_list_from_multiple_webpages(bibliography_file)
