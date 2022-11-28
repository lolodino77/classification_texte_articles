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

    def __init__(self, url, topic, num_articles):
        """ Constructor of DataSource """
        self.url = url
        self.topic = topic
        self.filename = self.create_corpus_txt_filename()
        self.path_corpus = "./data/input/corpus_txt/" + self.create_corpus_txt_filename() #self.filename
        self.num_articles = num_articles
        self.all_articles = self.create_all_articles()
        self.articles_urls = [] # will be defined in child method


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant l'article """
        print("str :")
        str_url = str(self.url)
        # str_articles_list = str(self.articles_list)
        str_filename = str(self.filename)
        str_path_corpus = str(self.path_corpus)
        str_num_articles = str(self.num_articles)
        str_all_articles = str(self.all_articles)
        desc = "url = "+ str_url 
        desc = desc + "\nfilename = " + str_filename
        desc = desc + "\npath_corpus = " + str_path_corpus
        desc += "\nnum_articles = " + str_num_articles
        desc += "\nall_articles = " + str_all_articles
        return(desc)


    def create_corpus_txt_filename(self):
        filename = "corpus_{}.txt".format(self.topic) 
        return(filename)


    def create_all_articles(self):
        """ Recupere la variable all_articles depuis la commande de terminal """
        if(self.num_articles == "all"):
            all_articles = True
        else:
            all_articles = False

        return(all_articles)
        

    def save_paragraphs(self, savemode="overwrite"):
        """Cree un corpus d'un topic (au format de liste de documents/textes) dans le fichier texte filename_output
        a partir d'une liste d'adresses urls d'articles

        Parametres: 
        articles_urls (liste de string) : La liste d'urls d'articles dont on veut extraire les paragraphes. 
                                            Ex : https://parlafoi.fr/lire/series/commentaire-de-la-summa/
        path_articles_list (string) : La liste des paths des listes d'articles
        path_corpus (string) : Le path vers le corpus, exemple = 
        save_mode (string) : Le mode d'ecriture du fichier ("append" = ajouter ou "overwrite" = creer un nouveau)
        
        Sortie:
        None : Fichier filename_corpus qui contient le corpus, une suite de textes separes par une saut de ligne

        Done : version "overwrite" recreer le corpus a chaque fois de zero 
        To do : version "append" ajouter du texte a un corpus deja cree, version "ignore" ne fais rien si fichier existe deja
                version "error" qui renvoie une erreur si fichier existe deja
        """
        #Ecrit dans le fichier texte filename_corpus.txt tous les paragraphes tous les articles d'une liste
        if(not self.all_articles):
            self.articles_urls = self.articles_urls[:self.num_articles] # garder que les num_articles premiers articles
            # rajouter cas ou il n'y a qu'un seul article
        if(savemode == "overwrite"):
            # print("path_corpus =", path_corpus)
            # print("articles_urls =", articles_urls)
            article_url = self.articles_urls[0]
            article = Article(article_url, self.topic)
            print("article_url =")
            print(article_url)
            # paragraphs = get_paragraphs_of_article(article_url)
            article.save_paragraphs(self.path_corpus, file_open_mode="w", sep = "\n\n")
            # save_paragraphs(paragraphs, path_corpus, file_open_mode="w", sep = "\n\n")
            for article_url in self.articles_urls[1:]:
                article = Article(article_url, self.topic)
                article.save_paragraphs(self.path_corpus, file_open_mode="a", sep = "\n\n")

                print("article_url =")
                print(article_url)
                # paragraphs = get_paragraphs_of_article(article_url)
                # save_paragraphs(paragraphs, path_corpus, file_open_mode="a", sep = "\n\n")
        elif(savemode == "append"):
            for article_url in self.articles_urls:
                print("article_url =")
                print(article_url)
                article = Article(article_url, self.topic)
                article.save_paragraphs(self.path_corpus, file_open_mode="a", sep = "\n\n")
