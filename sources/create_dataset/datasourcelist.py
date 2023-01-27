from datasource import *


class DataSourceList:
    """ Represente une liste d'objets DdataSources """

    def __init__(self, filename, num_articles, table_format):
        """ Constructeur de la classe Datasourcelist """
        self.filename = filename
        self.urls = "" #soit une liste d'urls bibliographies soit liste de blogs
        self.dataSources = []
        self.num_articles = num_articles
        self.table_format = table_format


    def __str__(self):
        """ Descripteur de la classe Datasourcelist """
        str_urls = str(self.urls)
        str_dataSources = str(self.dataSources)
        str_num_articles = str(self.num_articles)  
        desc = "urls = "+ str_urls 
        desc += "\ndataSources = " + str_dataSources
        self.print_datasources()
        desc += "\nnum_articles = " + str_num_articles

        return(desc)


    def print_datasources(self):
        """ Affiche la liste d'objets DataSource (soit une liste de blogs soit une liste de bibliographies) """
        for dataSource in self.dataSources:
            print("\n")
            print(dataSource)
        print("\n")


    def save_corpus_txt(self):
        """ Ecrit dans un fichier .txt les paragraphes de chaque DataSource de la liste """
        dataSource = self.dataSources[0]
        dataSource.save_corpus_txt(savemode="overwrite")        
        for dataSource in self.dataSources[1:]:
            dataSource.save_corpus_txt(savemode="append")


    def save_corpus_dataframe(self):
        """ Sauvegarde un corpus dans un dataframe (csv ou parquet) a partir d'un corpus au format .txt (
            une liste de messages
        ) """
        for dataSource in self.dataSources:
            dataSource.save_corpus_dataframe(self.table_format)