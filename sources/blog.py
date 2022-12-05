from datasource import *


class Blog(DataSource):
    def __init__(self, url, num_articles):
        DataSource.__init__(self, url, num_articles)
        print("type num_articles =", type(num_articles))
        self.corpus_name = self.get_corpus_name()
        self.path_corpus_txt = "./data/input/corpus_txt/" + self.create_corpus_txt_filename() #self.filename
        self.path_articles_urls = "./data/input/articles_lists/articles_list_{}.txt".format(self.corpus_name)


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant la bibliographie """
        print("str :")
        str_path_articles_urls = str(self.path_articles_urls)
        str_corpus_name = str(self.corpus_name)
        desc = DataSource.__str__(self)
        desc += "\ncorpus_name = " + str_corpus_name
        desc += "\npath_articles_urls = " + str_path_articles_urls
        return(desc)   


    def get_corpus_name(self):
        corpus_name = self.url.split("//")[1].split(".")[0]
        return(corpus_name)


    def get_web_page_text_contents(self, url):
        """ Donne dans une string le contenu texte d'une page web simple (avec que du textec comme un fichier texte) """
        print("get contents of page web :", url)
        page = requests.get(url) #page.text donne le contenu texte d'un page web (comme si c'etait un fichier txt)    
        text_contents = page.text

        return(text_contents)


    def get_blog_robots_page(self):
        """ Recupere la page robots.txt d'un blog 
            Exemple : blog_name = "http://alexanderpruss.blogspot.com"
        """
        return(self.url + "/robots.txt")


    def get_sitemap_page(self):
        """ Recupere la page sitemap d'un blog """
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


    # def get_sitemap_from_main_sitemap(self, sitemap_page):
    #     """ Donne la liste des urls sitemap presents sur la principale page sitemap d'un blog """
    #     urls = self.get_web_page_text_contents(sitemap_page)
    #     urls = urls.replace('<?xml version=\'1.0\' encoding=\'UTF-8\'?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap>', "")
    #     urls = urls.replace("</loc></sitemap><sitemap>", "\n")
    #     urls = urls.replace("</loc></sitemap></sitemapindex>", "")
    #     urls = urls.replace("<loc>", "")
    #     urls = urls.split("\n")
    #     return(urls)


    # def get_urls_from_one_sitemap_subpage(self, sitemap_subpage):
    #     """ Recupere les urls sur une seule sous-page sitemap """
    #     urls = self.get_web_page_text_contents(sitemap_subpage)
    #     urls = urls.replace("""<?xml version='1.0' encoding='UTF-8'?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">""", "")
    #     urls = urls.replace("</lastmod></url><url><loc>", "")
    #     urls = urls.replace("</loc><lastmod>", "\n")
    #     urls = re.sub(r"\d\d\d\d-\d\d-\d\d\w\d\d:\d\d:\d\d\w", "", urls, 0) #pour enlever les string comme 2022-10-21T00:02:02Z
    #     urls = urls.replace("<url><loc>", "\n")
    #     urls = urls.replace("\n</lastmod></url></urlset>", "\n")
    #     urls = urls.split("\n")[1:] #enlever premier element egal a ""
        
    #     return(urls)


    # def get_urls_from_all_sitemap_subpages(self, sitemap_subpages):
    #     """ Recupere les urls de toutes les sous-pages sitemap """
    #     urls = []
    #     for sitemap_subpage in sitemap_subpages:
    #         urls += self.get_urls_from_one_sitemap_subpage(sitemap_subpage)
        
    #     return(urls)


    # def create_articles_urls(self):
    #     """ Renvoie dans une liste tous les articles d'un blog (wordpress ou blogspot) a partir de sa page d'accueil
    #         Exemple : "https://majestyofreason.wordpress.com/", "https://edwardfeser.blogspot.com"
    #         Pour l'instant fonctionne que avec les blogs blogspot
    #     """
    #     # Recupere dans une liste urls les adresses url de tous les articles publies d'un blog
    #     sitemap_page = self.get_sitemap_page()
    #     sitemap_subpages = self.get_sitemap_from_main_sitemap(sitemap_page)
    #     urls = self.get_urls_from_all_sitemap_subpages(sitemap_subpages)
    #     urls = [url for url in urls if(len(url) > 1)] # enlever les ""
    #     urls = [url for url in urls if("pdf" not in url)] # enlever les articles pdf

    #     if(self.all_articles):
    #         self.num_articles = len(urls)

    #     print("in final function :")
    #     print("self.num_articles =", self.num_articles)
    #     print("type(self.num_articles) =", type(self.num_articles))
    #     urls = urls[:self.num_articles]

    #     print("check ''", "\n" in urls)

    #     return(urls)