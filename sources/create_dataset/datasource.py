import os
import glob
import requests
import re
from bs4 import BeautifulSoup
import html2text
from lib_general import *
from article import *


class DataSource:
    """ Define an article from a blog """

    def __init__(self, url, num_articles):
        """ Constructor of DataSource """
        self.url = url
        self.corpus_name = ""
        self.path_corpus_txt = ""
        self.filename_corpus_txt = ""
        self.num_articles = num_articles
        if(type(self.num_articles) != str):
            self.num_articles = int(self.num_articles)
        self.all_articles = self.create_all_articles()
        self.articles_urls = [] # will be defined in child method
        self.paragraphs = []


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant l'article """
        print("str :")
        str_url = str(self.url)
        # str_articles_list = str(self.articles_list)
        str_num_articles = str(self.num_articles)
        str_all_articles = str(self.all_articles)
        desc = "url = "+ str_url 
        desc += "\nnum_articles = " + str_num_articles
        desc += "\nall_articles = " + str_all_articles
        return(desc)


    def create_corpus_txt_filename(self):
        filename = "corpus_{}.txt".format(self.corpus_name) 
        return(filename)


    def save_articles_urls(self, file_open_mode="w", sep = "\n"):
        """ # Enregistrer la liste des urls d'articles """
        print("in save_articles_urls debut")
        save_list_to_txt(self.articles_urls, self.path_articles_urls, file_open_mode, sep)
        print("in save_articles_urls fin")


    def create_all_articles(self):
        """ Recupere la variable all_articles depuis la commande de terminal """
        if(self.num_articles == "all"):
            all_articles = True
        else:
            all_articles = False

        return(all_articles)


    def save_corpus_txt(self, savemode="overwrite"):
        """Cree un corpus d'un corpus_name (au format de liste de documents/textes) dans le fichier texte filename_output
        a partir d'une liste d'adresses urls d'articles

        Parametres: 
        articles_urls (liste de string) : La liste d'urls d'articles dont on veut extraire les paragraphes. 
                                            Ex : https://parlafoi.fr/lire/series/commentaire-de-la-summa/
        path_articles_urls (string) : La liste des paths des listes d'articles
        path_corpus_txt (string) : Le path vers le corpus, exemple = 
        save_mode (string) : Le mode d'ecriture du fichier ("append" = ajouter ou "overwrite" = creer un nouveau)
        
        Sortie:
        None : Fichier filename_corpus qui contient le corpus, une suite de textes separes par une saut de ligne

        Done : version "overwrite" recreer le corpus a chaque fois de zero 
        To do : version "append" ajouter du texte a un corpus deja cree, version "ignore" ne fais rien si fichier existe deja
                version "error" qui renvoie une erreur si fichier existe deja
        """
        # Garder que les num_articles premiers articles si on ne les stocke pastous 
        if(not self.all_articles):
            self.articles_urls = self.articles_urls[:self.num_articles] 
            # rajouter cas ou il n'y a qu'un seul article
        
        #Ecrit dans le fichier texte filename_corpus.txt tous les paragraphes tous les articles d'une liste
        if(savemode == "overwrite"):
            # print("path_corpus_txt =", path_corpus_txt)
            # print("articles_urls =", articles_urls)
            article_url = self.articles_urls[0] #premier article
            article = Article(article_url)
            print("article_url =")
            print(article_url)
            article.save_corpus_txt(self.path_corpus_txt, self.paragraphs, file_open_mode="w", sep = "\n\n")
            self.paragraphs.extend(article.paragraphs)

            for article_url in self.articles_urls[1:]: #tous les articles suivants
                article = Article(article_url)
                article.save_corpus_txt(self.path_corpus_txt, self.paragraphs, file_open_mode="a", sep = "\n\n")
                self.paragraphs.extend(article.paragraphs)
                print("len self.paragraphs =", len(self.paragraphs))
                print("article_url =")
                print(article_url)
        elif(savemode == "append"):
            for article_url in self.articles_urls:
                print("article_url =")
                print(article_url)
                article = Article(article_url)
                article.save_corpus_txt(self.path_corpus_txt, self.paragraphs, file_open_mode="a", sep = "\n\n")
                self.paragraphs.extend(article.paragraphs)
                print("len self.paragraphs =", len(self.paragraphs))
        #Enleve les doublons        
        paragraphs = open(self.path_corpus_txt, "r", encoding="utf-8").read().split("\n\n")
        print("avant retirer doublons =", len(paragraphs))
        paragraphs = list(set(paragraphs))
        print("apres retirer doublons =", len(paragraphs))


    def save_corpus_csv(self, df):
        """ Sauvegarde le corpus pandas dataframe dans un fichier csv """
        filename_corpus_csv = self.filename_corpus_txt.split(".txt")[0] + ".csv"
        df.to_csv("./data/input/corpus_csv/" + filename_corpus_csv, index=False)


    def save_corpus_parquet(self, df):
        """ Sauvegarde le corpus pandas dataframe dans un fichier parquet"""
        filename_corpus_parquet = self.filename_corpus_txt.split(".txt")[0] + ".parquet"
        df.to_parquet("./data/input/corpus_parquet/" + filename_corpus_parquet)


    def save_corpus_dataframe(self, format):
        """ Sauvegarde un corpus dans un dataframe (csv ou parquet) a partir d'un corpus dans un .txt """
        print("in save_corpus_dataframe")
        print("filename_corpus_txt =", self.filename_corpus_txt)
        res = open("./data/input/corpus_txt/" + self.filename_corpus_txt, "r", encoding="utf-8").read().split("\n\n")
        res = [elt for elt in res if len(elt) > 1]
        message = res
        length = [len(elt) for elt in res]
        list_of_rows = list(zip(message, length))
        df = pd.DataFrame(list_of_rows, columns=["message", "length"])

        if(format == "csv"):
            self.save_corpus_csv(df)
        elif(format == "parquet"):
            self.save_corpus_parquet(df)