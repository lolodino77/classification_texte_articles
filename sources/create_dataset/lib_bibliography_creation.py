import requests
import re


def get_web_page_text_contents(url):
    """ Donne dans une string le contenu texte d'une page web simple (avec que du textec comme un fichier texte) """
    page = requests.get(url) #page.text donne le contenu texte d'un page web (comme si c'etait un fichier txt)    
    text_contents = page.text

    return(text_contents)


def get_blog_robots_page(blog_name):
    return(blog_name + "/robots.txt")


def get_sitemap_page(blog_name):
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


def write_bibliography_of_an_entire_blog(blog_name, urls):
    """ Ecrit une bibliographie de tous les articles d'un blog donne (wordpress ou blogspot) a partir d'une
        liste d'urls
        Exemple : "https://majestyofreason.wordpress.com/", "https://edwardfeser.blogspot.com" 
    """
    author = get_author_from_blog_name(blog_name)
    bibliography_name = "./data/input/bibliographies/bibliography_{}.txt".format(author)
    f = open(bibliography_name, "w")
    for url in urls:
        f.write(url + "\n")
    f.close()