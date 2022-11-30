from datasourcelist import *
from blog import *

class BlogList(DataSourceList):
    def __init__(self, urls, num_articles):
        """ Constructor of Bloglist """
        DataSourceList.__init__(self, urls, num_articles)
        self.urls = self.define_urls(urls)
        self.define_datasources()

    # def __str__(self):
        

    def define_urls(self, urls):
        """ Definit urls, si besoin va chercher la liste urls dans un fichier texte """
        if(type(urls) == str and "http://" not in urls): #si urls = nom d'un fichier texte
            path = "./data/input/blogs/{}".format(urls)
            urls = open(path, "r", encoding="utf-8").read().split("\n")
        return(urls)

    def define_datasources(self):
        """ Definit la liste d'objets DataSources (liste de Bibliography ou de Blog """
        for url in self.urls:
            blog = Blog(url, self.num_articles)
            self.dataSources.append(blog)
            print("self.dataSources =", self.dataSources)

    def save_articles_urls(self):
        for blog in self.dataSources:
            blog.save_articles_urls()