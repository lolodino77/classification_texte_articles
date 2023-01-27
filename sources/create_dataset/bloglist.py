from datasourcelist import *
from blog import *
from blogspot import *
from wordpress import *


class BlogList(DataSourceList):
    def __init__(self, filename, num_articles, table_format):
        """ Constructeur de la classe Bloglist """
        DataSourceList.__init__(self, filename, num_articles, table_format)
        self.urls = self.define_urls()
        self.define_datasources()

    def __str__(self):
        """ Descripteur de la classe Bloglist """
        desc = DataSourceList.__str__(self)
        return(desc)

    def define_urls(self):
        """ Definit l'attribut urls (une liste d'urls de blogs) a partir d'un fichier .txt qui les contient 
        d'urls dans un fichier texte """        
        path = "./data/input/blogs/{}".format(self.filename)
        urls = open(path, "r", encoding="utf-8").read().split("\n")
        return(urls)

    def define_datasources(self):
        """ Definit l'attribut datasources (la liste d'objets DataSources), ici une liste d'objets Blog """
        for url in self.urls:
            if("blogspot" in url):
                print("c'est blogspot")
                blog = Blogspot(url, self.num_articles)
                self.dataSources.append(blog)
            elif("wordpress" in url):
                print("c'est wordpress")
                blog = Wordpress(url, self.num_articles)
                self.dataSources.append(blog)
            print("self.dataSources =", self.dataSources)

    def save_articles_urls(self):
        """ Pour chaque blog, ecrit l'attribut articles_urls (des articles d'un blog, tous ou un nombre defini)
        dans un fichier .txt different """
        for blog in self.dataSources:
            blog.save_articles_urls()