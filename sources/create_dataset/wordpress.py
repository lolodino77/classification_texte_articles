from blog import *


class Wordpress(Blog):

    def __init__(self, url, num_articles):
        Blog.__init__(self, url, num_articles)
        self.articles_urls = self.create_articles_urls()
        

    def __str__(self):
        desc = Blog.__str__(self)
        str_articles_urls = str(self.articles_urls)
        desc += "\narticles_urls = " + str_articles_urls

        return(desc)


    def create_articles_urls(self):
        """ Renvoie dans une liste tous les articles d'un blog (wordpress ou blogspot) a partir de sa page d'accueil
            Exemple : "https://majestyofreason.wordpress.com/", "https://edwardfeser.blogspot.com"
            Pour l'instant fonctionne que avec les blogs blogspot
        """
        # Recupere dans une liste urls les adresses url de tous les articles publies d'un blog
        sitemap_page = self.get_sitemap_page()
        sitemap_page_contents = self.get_web_page_text_contents(sitemap_page)
        urls = re.findall("<loc>(.*?)</loc>", sitemap_page_contents)
        # urls = [url for url in urls if(len(url) > 1)] # enlever les ""
        # urls = [url for url in urls if("pdf" not in url)] # enlever les articles pdf

        if(self.all_articles):
            self.num_articles = len(urls)

        print("in final function :")
        print("self.num_articles =", self.num_articles)
        print("type(self.num_articles) =", type(self.num_articles))
        urls = urls[:self.num_articles]

        print("check ''", "\n" in urls)

        return(urls)