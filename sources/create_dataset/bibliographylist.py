from datasourcelist import *
from bibliography import *

class BibliographyList(DataSourceList):
    def __init__(self, filename, num_articles, table_format):
        """ Constructor of Bibliographylist """
        DataSourceList.__init__(self, filename, num_articles, table_format)
        # self.dataSources = self.define_datasources()
        self.urls = self.define_urls()
        self.corpus_name = self.define_corpus_name()
        self.define_datasources()
        
    def define_corpus_name(self):
        corpus_name = self.filename.split("bibliography_")[1].split(".txt")[0]
        return(corpus_name)

    def define_urls(self):
        """ Definit urls, si besoin va chercher la liste urls dans un fichier texte """
        # if(type(urls) == str and "http://" not in urls): #si urls = nom d'un fichier texte
        path = "./data/input/bibliographies/{}".format(self.filename)
        urls = open(path, "r", encoding="utf-8").read().split("\n")
        return(urls)

    def define_datasources(self):
        """ Definit la liste d'objets DataSources (liste de Bibliography ou de Blog """
        for url in self.urls:
            bibliography = Bibliography(url, self.corpus_name, self.num_articles)
            self.dataSources.append(bibliography)

    def save_articles_urls(self):
        bibliography = self.dataSources[0]
        bibliography.save_articles_urls(file_open_mode="w", sep = "\n")
        for bibliography in self.dataSources[1:]:
            bibliography.save_articles_urls(file_open_mode="a", sep = "\n")