from datasource import *


class DataSourceList:
    """ Define a list of datasources """

    def __init__(self, filename, num_articles):
        """ Constructor of Datasourcelist """
        self.filename = filename
        self.urls = "" #soit liste de bibliographies soit liste de blogs
        self.dataSources = []
        self.num_articles = num_articles

    def __str__(self):
        str_urls = str(self.urls)
        str_dataSources = str(self.dataSources)
        str_num_articles = str(self.num_articles)  
        desc = "urls = "+ str_urls 
        desc += "\ndataSources = " + str_dataSources
        self.print_datasources()
        desc += "\nnum_articles = " + str_num_articles

        return(desc)

    def print_datasources(self):
        for dataSource in self.dataSources:
            print("\n")
            print(dataSource)
        print("\n")

    def save_paragraphs(self):
        """ Ecrit les paragraphes de chaque DataSource de la liste dans un fichier texte"""
        dataSource = self.dataSources[0]
        print("dataSource =", dataSource)
        # dataSource.save_articles_urls()
        dataSource.save_paragraphs(savemode="overwrite")        
        for dataSource in self.dataSources[1:]:
            print("dataSource =", dataSource)
            # dataSource.save_articles_urls()
            dataSource.save_paragraphs(savemode="append")