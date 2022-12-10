from lib_general import *
import nltk
import re
import string
import pandas as pd


class MultipleTopicsCorpus:
    """ Definit un corpus qui contient des documents d'un seul topic """

    def __init__(self, corpus_lists_filename, topics_list_filename):
        """ Constructor of Article 
        filename_input (string) : nom du fichier qui contient les noms des fichiers .txt des corpus a fusionner
        corpus_to_merge_filenames (liste de string) : liste des fichiers .txt des corpus a fusionner
        corpus_dataframes (liste de pandas dataframes) : liste de pandas dataframes 
        
        corpus_lists_filename (string) : un fichier .txt contenant les fichiers "liste de corpus txt" pour chaque topic 
        topics_list_filename (string) : un fichier .txt contenant la liste des topics correspondant a chaque liste de corpus
        """
        self.corpus_to_merge_filenames = corpus_lists_filename
        self.topics_list_filename = topics_list_filename
        self.corpus_to_merge = self.create_corpus_to_merge() 
        self.topics = self.create_topics()
        # self.filename_input = filename_input
        self.corpus_txt_filenames = self.create_corpus_txt_filenames()
        self.corpus_dataframes = self.create_corpus_dataframes()
        self.corpus_merged_dataframe = self.create_corpus_merged_dataframe()


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant l'article """
        print("str :")
        str_filename_input = str(self.filename_input)
        str_corpus_txt_filenames = str(self.corpus_txt_filenames)
        str_corpus_dataframes = str(self.corpus_dataframes)
        str_corpus_merged_dataframe = str(self.corpus_merged_dataframe)
        str_topic = str(self.topic)
        desc = "str_filename_input = " + str_filename_input
        desc += "\nstr_topic = " + str_topic
        desc += "\nstr_corpus_txt_filenames = " + str_corpus_txt_filenames
        desc += "\ncorpus_dataframes = " + str_corpus_dataframes
        desc += "\n\ncorpus_merged_dataframe = " + str_corpus_merged_dataframe
        return(desc)