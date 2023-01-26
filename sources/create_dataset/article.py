import sys
sys.path.append("../")

import requests
from bs4 import BeautifulSoup
import html2text
from lib_general import *


class Article:
    """ Define an article/a post from a blog """

    def __init__(self, url):
        """ Constructeur de la classe Article """
        self.url = url
        self.paragraphs = self.create_paragraphs()


    def __str__(self):
        """ Descripteur de la classe Article """
        print("str :")
        str_url = str(self.url)
        str_paragraphs = str(self.paragraphs)
        desc = "url = "+ str_url
        desc += "\nparagraphs = " + str_paragraphs 

        return(desc)


    def create_paragraphs(self):
        """ Renvoie les paragraphes d'un article dans une liste
    
        Sortie:
            paragraphs (list of str) : Tous les paragraphes d'un article
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
        paragraphs = txt.split("\n\n") 

        #Enleve les doublons
        paragraphs = list(set(paragraphs))
        
        stop_words = ["Email", "(javascript:void\(0\))", "Creative Commons License", "Simple theme", "www.blogger", "profile"]
        paragraphs = [paragraphe for paragraphe in paragraphs if(all(word not in paragraphe for word in stop_words))]

        #Enleve les paragraphes avec des phrases trop courtes (trop peu de mots)
        paragraphs = [paragraphe for paragraphe in paragraphs if len(paragraphe.split(" ")) > 12]
        
        #Enleve les paragraphes qui contiennent # (car pas des paragraphes)
        paragraphs = [paragraphe for paragraphe in paragraphs if("#" not in paragraphe)]

        #Enleve les paragraphes avec trop peu de caracteres
        paragraphs = [paragraphe for paragraphe in paragraphs if(len(paragraphe) > 12)]

        return(paragraphs)


    def save_corpus_txt(self, path_corpus_txt, corpus_paragraphs="", file_open_mode="w", sep = "\n\n"):
        """ Sauvegarde les paragraphes d'un article dans un fichier texte """
        f = open(path_corpus_txt, file_open_mode, encoding="utf-8") #"w" si n'existe pas, "a" si on veut ajouter a un fichier deja existant
        for paragraph in self.paragraphs:
            if(paragraph not in corpus_paragraphs):
                f.write(paragraph + sep)
        f.close()