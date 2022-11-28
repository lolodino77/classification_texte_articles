import os
import glob
import requests
import re
from bs4 import BeautifulSoup
import html2text
from lib_general import *


class Article:
    """ Define an article from a blog """

    def __init__(self, url, topic):
        """ Constructor of Article """
        self.url = url
        self.paragraphs = self.create_paragraphs()


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant l'article """
        print("str :")
        str_url = str(self.url)
        # str_articles_list = str(self.articles_list)
        str_paragraphs = str(self.paragraphs)
        desc = "url = "+ str_url
        desc += "\nparagraphs = " + str_paragraphs 
        # desc = desc + "\nfilename = " + str_filename
        return(desc)   


    def create_paragraphs(self):
        """ Renvoie les paragraphes d'un article dans une liste
        
        Parametres: 
        article_url (string) : L'url de l'article a decouper en plusieurs parties
        
        Sortie:
        None : Fichier output_filename qui contient les documents de l'article dont l'url est article_url
        """
        # Recupere le texte de la page web a l'aide d'un parser
        # Recupere le texte d'un article mais avec des balises html (<\p><p> par exemple)
        page = requests.get(url=self.url)
        soup = BeautifulSoup(page.content, 'html.parser')
        txt = str(soup) 

        # Conversion des indicateurs de paragraphes et de sections /p et /li en retours a la ligne \n pour le split
        txt = txt.replace("\n", " ")
        txt = txt.replace("</p>", "</p>\n\n")
        txt = txt.replace("<li>", "<p>")
        txt = txt.replace("</li>", "</p>\n\n")
        
        # Suppression des balises html
        txt = html2text.html2text(txt)

        # Decoupage en plusieurs parties avec pour separateur le retour a la ligne \n
        txt = txt.split("\n\n") 
        # print("txt")
        # print(txt)

        #Enleve les paragraphes avec trop peu de caracteres
        paragraphs = [paragraphe for paragraphe in txt if len(paragraphe) > 12] 
        
        #Enleve les paragraphes avec des phrases trop courtes (trop peu de mots)
        paragraphs = [paragraphe for paragraphe in txt if len(paragraphe.split(" ")) > 10]
        
        return(paragraphs)


    def save_paragraphs(self, path_corpus, file_open_mode="w", sep = "\n\n"):
        """ Sauvegarde les paragraphes d'un article dans un fichier texte """
        print("file_open_mode =", file_open_mode)
        save_list_to_txt(self.paragraphs, path_corpus, file_open_mode, sep)