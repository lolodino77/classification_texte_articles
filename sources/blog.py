from datasource import *


class Blog(DataSource):
    def __init__(self, url, topic, num_articles):
        DataSource.__init__(self, url, topic)
        self.name = self.topic
        self.articles_urls = self.create_articles_urls()


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant la bibliographie """
        print("str :")
        str_articles_urls = str(self.articles_urls)
        str_num_articles = str(self.num_articles)
        str_all_articles = str(self.all_articles)
        desc = DataSource.__str__(self)
        desc += "\narticles_urls = " + str_articles_urls
        desc += "\nnum_articles = " + str_num_articles
        desc += "\nall_articles = " + str_all_articles
        return(desc)   


    def get_web_page_text_contents(self):
        """ Donne dans une string le contenu texte d'une page web simple (avec que du textec comme un fichier texte) """
        page = requests.get(self.url) #page.text donne le contenu texte d'un page web (comme si c'etait un fichier txt)    
        text_contents = page.text

        return(text_contents)


    def get_blog_robots_page(self):
        """ Recupere la page robots.txt d'un blog 
            Exemple : blog_name = "http://alexanderpruss.blogspot.com"
        """
        return(self.name + "/robots.txt")


    def get_sitemap_page(self):
        """ Recupere la page sitemap d'un blog """
        robots_txt_page = self.get_blog_robots_page(self.name)
        text_contents = self.get_web_page_text_contents(self, robots_txt_page)
        text_contents = text_contents.split("\n")
        sitemap_contents = [elt for elt in text_contents if "Sitemap" in elt]
        print("sitemap_contents =", sitemap_contents)
        sitemap_page = sitemap_contents[0].split(" ")[1]

        return(sitemap_page)


    def get_sitemap_from_main_sitemap(self, sitemap_page):
        """ Donne la liste des urls sitemap presents sur la principale page sitemap d'un blog """
        urls = self.get_web_page_text_contents(sitemap_page)
        urls = urls.replace('<?xml version=\'1.0\' encoding=\'UTF-8\'?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap>', "")
        urls = urls.replace("</loc></sitemap><sitemap>", "\n")
        urls = urls.replace("</loc></sitemap></sitemapindex>", "")
        urls = urls.replace("<loc>", "")
        urls = urls.split("\n")
        return(urls)


    def get_urls_from_one_sitemap_subpage(self, sitemap_subpage):
        """ Recupere les urls sur une seule sous-page sitemap """
        urls = self.get_web_page_text_contents(self, sitemap_subpage)
        urls = urls.replace("""<?xml version='1.0' encoding='UTF-8'?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">""", "")
        urls = urls.replace("</lastmod></url><url><loc>", "")
        urls = urls.replace("</loc><lastmod>", "\n")
        urls = re.sub(r"\d\d\d\d-\d\d-\d\d\w\d\d:\d\d:\d\d\w", "", urls, 0) #pour enlever les string comme 2022-10-21T00:02:02Z
        urls = urls.replace("<url><loc>", "\n")
        urls = urls.replace("\n</lastmod></url></urlset>", "\n")
        urls = urls.split("\n")[1:] #enlever premier element egal a ""
        
        return(urls)


    def get_urls_from_all_sitemap_subpages(self, sitemap_subpages):
        """ Recupere les urls de toutes les sous-pages sitemap """
        urls = []
        for sitemap_subpage in sitemap_subpages:
            urls += self.get_urls_from_one_sitemap_subpage(sitemap_subpage)
        
        return(urls)


    def create_articles_urls(self):
        """ Renvoie dans une liste tous les articles d'un blog (wordpress ou blogspot) a partir de sa page d'accueil
            Exemple : "https://majestyofreason.wordpress.com/", "https://edwardfeser.blogspot.com"
            
            Entrees:
            self.name (string) : Le nom du blog
            keep_all_articles (booleen) : Indique si on recupere tous les articles ou seulement un nombre limite
            num_articles (int) : Le nombre d'articles a garder (valeur defaut 0 mais n'importe quelle valeur possible)
                                Si keep_all_articles = True, pas besoin de preciser la nombre d'articles a garder num_articles
            Sorties:
            urls (liste de string) : exemple=["www.aaa.wordpress.fr\n", "www.aba.wordpress.fr\n", "www.aaa.wordpress.fr"] 
        """
        # Recupere dans une liste urls les adresses url de tous les articles publies d'un blog
        sitemap_page = self.get_sitemap_page()
        sitemap_subpages = self.get_sitemap_from_main_sitemap(sitemap_page)
        urls = self.get_urls_from_all_sitemap_subpages(sitemap_subpages)
        urls = [url for url in urls if(len(url) > 1)] # enlever les ""
        urls = [url for url in urls if("pdf" not in url)] # enlever les articles pdf

        # if(keep_all_articles):
        #     num_articles = len(urls)
        urls = urls[:self.num_articles]

        print("check ''", "\n" in urls)

        return(urls)