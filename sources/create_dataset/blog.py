from datasource import *


class Blog(DataSource):
    """ Represente un blog avec son nom, ses articles, etc. """
    def __init__(self, url, num_articles):
        DataSource.__init__(self, url, num_articles)
        print("type num_articles =", type(num_articles))
        self.corpus_name = self.get_corpus_name()
        print("corpus_name ==", self.corpus_name)
        self.filename_corpus_txt = self.create_corpus_txt_filename()
        self.path_corpus_txt = "./data/input/corpus_txt/" + self.filename_corpus_txt 
        self.path_articles_urls = "./data/input/articles_lists/articles_list_{}.txt".format(self.corpus_name)


    def __str__(self):
        """ Descripteur de la classe Blog """
        print("str :")
        str_path_articles_urls = str(self.path_articles_urls)
        str_corpus_name = str(self.corpus_name)
        desc = DataSource.__str__(self)
        desc += "\ncorpus_name = " + str_corpus_name
        desc += "\npath_articles_urls = " + str_path_articles_urls
        return(desc)   


    def get_corpus_name(self):
        """ Retourne le nom du corpus correspondant au blog (paragraphes de ses articles) 
        Sortie :
            corpus_name (string) : Le nom du corpus correspondant au blog
        """
        corpus_name = self.url.split("//")[1].split(".")[0]
        return(corpus_name)


    def get_web_page_text_contents(self, url):
        """ Retourne le contenu texte d'une page web simple dans une string 
        Entree :
            url (string) : L'url de la page
        Sortie :
            text_contents (string) : Le contenu texte d'une page web simple
        """
        page = requests.get(url) #page.text donne le contenu texte d'un page web (comme si c'etait un fichier txt)    
        text_contents = page.text

        return(text_contents)


    def get_blog_robots_page(self):
        """ Recupere la page robots.txt du blog 
        Sortie :
            url_robots_txt (string) : L'url de la page robots du blog
            Exemple : http://something.blogspot.com => http://something.blogspot.com/robots.txt
        """
        url_robots_txt = self.url + "/robots.txt"
        
        return(url_robots_txt)


    def get_sitemap_page(self):
        """ Recupere la page sitemap du blog
        Sortie :
            sitemap_page (string) : L'url de la page sitemap du blog
            Exemple : http://alexanderpruss.blogspot.com => http://alexanderpruss.blogspot.com/robots.txt
        """
        robots_txt_page = self.get_blog_robots_page()
        robots_txt_contents = self.get_web_page_text_contents(robots_txt_page)

        # Cas 1 : il y a une page robots.txt sur le blog (liste des "pages autorisees")
        if(robots_txt_contents != ""):
            robots_txt_contents = robots_txt_contents.split("\n")
            sitemap_contents = [elt for elt in robots_txt_contents if "Sitemap" in elt]
            sitemap_page = sitemap_contents[0].split(" ")[1]

        # Cas 2 : pas de page robots.txt sur le blog (liste des "pages autorisees")
        else:
            sitemap_page = self.url + "/sitemap.xml"

        return(sitemap_page)