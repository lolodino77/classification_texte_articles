from datasourcelist import *
from blog import *

class BlogList(DataSourceList):
    def __init__(self, filename, num_articles):
        """ Constructor of Bloglist """
        DataSourceList.__init__(self, filename, num_articles)
        self.urls = self.define_urls()
        self.define_datasources()

    # def __str__(self):
        

    def define_urls(self):
        path = "./data/input/blogs/{}".format(self.filename)
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