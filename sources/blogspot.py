from blog import *


class Blogspot(Blog):

    def __init__(self, url, num_articles):
        Blog.__init__(self, url, num_articles)
        self.articles_urls = self.create_articles_urls()
        

    def __str__(self):
        desc = Blog.__str__(self)
        str_articles_urls = str(self.articles_urls)
        desc += "\narticles_urls = " + str_articles_urls

        return(desc)


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
        print("in get_urls_from_one_sitemap_subpage")
        urls = self.get_web_page_text_contents(sitemap_subpage)
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
            Pour l'instant fonctionne que avec les blogs blogspot
        """
        # Recupere dans une liste urls les adresses url de tous les articles publies d'un blog
        sitemap_page = self.get_sitemap_page()
        print("in create_articles_urls")
        print("sitemap_page =", sitemap_page)
        sitemap_subpages = self.get_sitemap_from_main_sitemap(sitemap_page)
        print("len sitemap_subpages =", len(sitemap_subpages))

        # Cas 1 : ou il y a des sous-pages sitemap
        if(len(sitemap_subpages) > 1):
            urls = self.get_urls_from_all_sitemap_subpages(sitemap_subpages)
                    
        # Cas 2 : ou il n'y qu'une seule page sitemap (pas de sous-pages)
        else:
            urls = self.get_urls_from_one_sitemap_subpage(sitemap_page)

        urls = [url for url in urls if(len(url) > 1)] # enlever les ""
        urls = [url for url in urls if("pdf" not in url)] # enlever les articles pdf

        if(self.all_articles): #definit le nombre d'articles a "scraper"
            self.num_articles = len(urls)

        print("in final function :")
        print("self.num_articles =", self.num_articles)
        print("type(self.num_articles) =", type(self.num_articles))
        urls = urls[:self.num_articles]

        print("check ''", "\n" in urls)

        return(urls)