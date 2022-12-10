from datasourcelist import *
from blog import *
from blogspot import *
from wordpress import *


class BlogList(DataSourceList):
    def __init__(self, filename, num_articles, table_format):
        """ Constructor of Bloglist """
        DataSourceList.__init__(self, filename, num_articles, table_format)
        self.urls = self.define_urls()
        self.define_datasources()

    def __str__(self):
        desc = DataSourceList.__str__(self)
        return(desc)

    def define_urls(self):
        path = "./data/input/blogs/{}".format(self.filename)
        urls = open(path, "r", encoding="utf-8").read().split("\n")
        return(urls)

    def define_datasources(self):
        """ Definit la liste d'objets DataSources (liste de Bibliography ou de Blog """
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
        for blog in self.dataSources:
            blog.save_articles_urls()