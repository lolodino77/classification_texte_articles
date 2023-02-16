import os
import sys
sys.path.insert(0, "..")
from lib_general import *
import pandas as pd
import numpy as np
import os
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from sklearn.preprocessing import LabelEncoder
import joblib


class CorpusMerger:
    """ Classe pour fusionner plusieurs corpus (a un seul topic) en un seul corpus (a plusieurs topics)
    1. Soit un corpus final avec un seul et unique topic
    2. Soit un corpus final avec plusieurs topics
    """

    def __init__(self, filename, language):
        """ Constructeur de la classe CorpusMerger 
        1er constructeur : prend plusieurs corpus à un seul topic (au moins 2 fichiers different)
            corpus_txt_list_filename : un fichier .txt dans ce cas 
        2eme constructeur : prend un seul corpus à deux topics (1 seul fichier)
            corpus_txt_list_filename : un fichier .csv ou .parquet dans ce cas 
        """
        self.input_file_extension = get_file_extension(filename)
        if(self.input_file_extension == "txt"): #plusieurs corpus a un topic chacun en entrees
            two_classes_input = False
            self.language = language
            self.corpus_txt_list_filename = filename
            self.corpus_txt_filenames, self.topics = self.create_corpus_txt_filenames_and_topics() 
            self.topics_names_concat = "_".join(sorted(set(self.topics)))
            self.corpus_dataframes = self.create_corpus_dataframes()
            self.merged_corpus_name = self.create_merged_corpus_name(two_classes_input)
            self.merged_corpus_dataframe = self.create_merged_corpus_dataframe()
        if(self.input_file_extension == "csv" or self.input_file_extension == "parquet"): #un seul corpus a plusieurs topic en entree
            print("input_file_extension =", self.input_file_extension)
            two_classes_input = True 
            self.language = language
            # self.topics_names_concat = "_".join(sorted(set(self.topics)))
            self.merged_corpus_name = self.create_merged_corpus_name(two_classes_input, filename)
            self.merged_corpus_dataframe = get_raw_merged_corpus_from_filename(filename)

    def __str__(self):
        """ Descripteur de la classe CorpusMerger """
        str_corpus_txt_list_filename = str(self.corpus_txt_list_filename)
        str_corpus_txt_filenames = str(self.corpus_txt_filenames)
        str_topics = str(self.topics)
        str_corpus_dataframes = str(self.corpus_dataframes)
        str_merged_corpus_dataframe = str(self.merged_corpus_dataframe)
        desc = "corpus_txt_list_filename = " + str_corpus_txt_list_filename
        desc += "\ncorpus_txt_filenames = " + str_corpus_txt_filenames
        desc += "\ntopics = " + str_topics
        desc += "\ncorpus_dataframes = " + str_corpus_dataframes
        desc += "\n\nmerged_corpus_dataframe = " + str_merged_corpus_dataframe

        return(desc)


    def create_merged_corpus_name(self, two_classes_input=True, filename=""):
        """ Renvoie le nom de merged_corpus 
                1) Soit a partir des noms des corpus fusionnes (si plusieurs corpus a un topic en entrees)
                2) Soit a partir du nom du corpus (si un seul corpus a deux topics en entree) 
        Entree :
            two_classes_input (bool) : True si un seul corpus en entree deja avec deux classes, False sinon
            filename (str) : Nom du fichier donne en entree si corpus a deux classes en entree
        Sortie :
            merged_corpus_name (str) : le nom du merged_corpus
        """
        if(two_classes_input): #si on a en entree un corpus a deux topics
            merged_corpus_name = get_corpus_name_from_filename(filename)
        else: #si on a en entree plusieurs corpus a un topic chacun
            merged_corpus_name = self.topics_names_concat #ex : corpus avec topic chien + corpus avec topic chat = corpus_chien_chat

        return(merged_corpus_name)


    def create_corpus_txt_filenames_and_topics(self):
        """ Topics definis a partir du fichier corpus_txt_list_filename """
        f = open("./data/input/corpus_txt/" + self.corpus_txt_list_filename, "r",
                    encoding="utf-8")
        corpus_txt_filenames = []
        topics = []
        lines = f.read().splitlines()
        for line in lines:
            splitted_line = line.split(" ")
            corpus_txt_filenames.append(splitted_line[0])
            topics.append(splitted_line[1])
        
        return(corpus_txt_filenames, topics)
        

    def get_corpus_dataframe(self, corpus_txt_filename, topic):
        """ Renvoie un pandas dataframe a partir d'un corpus au format .txt 
        Entrees : 
            corpus_txt_filename (str) : nom du fichier du corpus au format .txt
            topic (str) : nom de la classe qu'on annotera a chaque message du corpus corpus_txt_filename
        Sortie :
            corpus_dataframe : le pandas dataframe correspondant
        """
        res = open("./data/input/corpus_txt/" + corpus_txt_filename, "r", encoding="utf-8").read().split("\n\n")
        res = [elt for elt in res if len(elt) > 1]
        message = res
        length = [len(elt) for elt in res]
        list_of_rows = list(zip(message, length))
        corpus_dataframe = pd.DataFrame(list_of_rows, columns=["message", "length"])
        corpus_dataframe["category"] = topic

        return(corpus_dataframe)


    def create_corpus_dataframes(self):
        """ Renvoie une liste de dataframes (chaque corpus) a partir des fichiers .txt 
        Sortie :
            all_corpus_dataframes (list of pandas dataframe) : la liste des pandas dataframe correspondants
        """
        all_corpus_dataframes = []
        for filename, topic in zip(self.corpus_txt_filenames, self.topics):
            corpus_dataframe = self.get_corpus_dataframe(filename, topic)
            all_corpus_dataframes.append(corpus_dataframe)
        
        return(all_corpus_dataframes)


    def create_merged_corpus_dataframe(self):
        """ Renvoie le dataframe final qui est la fusion de tous les dataframes 
        Sortie :
            merged_corpus (pandas dataframe) : la fusion de tous les dataframes
        """
        if(len(self.corpus_dataframes) > 1):
            merged_corpus = pd.concat(self.corpus_dataframes) 
        else:
            merged_corpus = self.corpus_dataframes[0]
        
        return(merged_corpus)
    

    def get_preprocessed_messages(self, lemmatizer, stopwords):
        """Nettoie tous les documents d'une liste pour creer un dataset exploitable par des modeles d'IA.
        Renvoie une liste avec tous les messages nettoyes (principalement pour des messages en francais)

        Parametres:
        lemmatizer (fonction) : Le lemmatizer qui servira a lemmatizer les mots des documents si possible
        stopwords (liste de string) : La liste des stopwords (mots frequents mais inutiles a enlever)

        Sortie:
        preprocess_list (liste de string) : Une liste de documents nettoyes
        """
        # cas speciaux restants a traiter :
        # mots avec un apostrophe avant (Traite)
        # mots composes avec un ou plusieurs tirets (A traiter)
        message_preprocessed = self.merged_corpus_dataframe["message"]

        #remplacer les virgules bizarres
        message_preprocessed = message_preprocessed.str.replace("’", "'")

        # # supprimer les mots avant les apostrophes (particules comme l', t', etc.)
        message_preprocessed = message_preprocessed.str.replace(r"\s\w+'", " ", regex=True)

        # # enlever la ponctuation et met en minuscule
        message_preprocessed.replace(to_replace=r"""['!"#$%&'()*+,./:;<=>?@[\]^_`{|}~]""", value='', inplace=True, regex=True)
        message_preprocessed = message_preprocessed.str.lower()

        # # enlever les chiffres
        message_preprocessed.replace(to_replace=r"[0-9]", value='', inplace=True, regex=True)

        # enlever les stopwords (mots n’apportant pas de sens) dans une column of string (pas liste de mots) -- 7 min
        message_preprocessed = message_preprocessed.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
        
        # transformer les phrases en liste de tokens (en liste de mots) -- 15 min
        message_preprocessed = message_preprocessed.apply(nltk.word_tokenize)

        # lemmatizer (convertir en la racine) et reformer la phrase en reliant les mots precedents
        message_preprocessed = message_preprocessed.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x]))
            
        return message_preprocessed 


    # preprocess_merged_corpus_dataframe vectorized, input two topics corpus for corpusmerger constructor

    # preprocess_merged_corpus_dataframe vectorized

    def preprocess_merged_corpus_dataframe(self):
        """ Nettoie le merged_corpus :
            1. Cree l'id unique
            2. Supprimme les colonnes inutiles
            3. Cree la colonne "len" taille de chaque document
            4. Cree la colonne "category_bin" annotations au format entier binaire (0, 1)
            5. Melange les lignes aleatoirement
            6. Supprime les retours a la ligne \n et \r
            7. Supprime les doublons (lignes qui ont le meme message)
        """
        # Recuperation du lemmatizer
        stopwords = nltk.corpus.stopwords.words(self.language)
        print("os.getcwd() =", os.getcwd())
        # mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))
        if(self.language == "french"):
            lemmatizer = FrenchLefffLemmatizer()
        elif(self.language == "english"):
            lemmatizer = WordNetLemmatizer() #le lemmatizer WordNetLemmatizer de nltk uniquement pour l'anglais 

        # Execution de la fonction principale qui fait le nettoyage
        self.merged_corpus_dataframe["message_preprocessed"] = self.get_preprocessed_messages(lemmatizer, stopwords)

        # 2. Suppression des colonnes inutiles
        self.merged_corpus_dataframe = self.merged_corpus_dataframe[["message", "message_preprocessed", "category"]]

        # 3. Creation de la taille de chaque documents (en nombre de caracteres)
        self.merged_corpus_dataframe["length"] = self.merged_corpus_dataframe["message"].str.len()
        self.merged_corpus_dataframe['length'] = self.merged_corpus_dataframe['length'].astype(int)

        # 4. Annotation au format entier (necessaire pour certaines fonctions de sklearn)
        # a. Cree une colonne category_bin avec les annotations au format entier binaire (0, 1)
        # b. Enregistre les labels et leur numero correspondant dans un dictionnaire
        # self.merged_corpus_dataframe["category_bin"] = np.select([self.merged_corpus_dataframe["category"] == class_1], [1], default=0)
        LE = LabelEncoder()
        self.merged_corpus_dataframe["category_bin"] = LE.fit_transform(self.merged_corpus_dataframe["category"]) 

        if(self.input_file_extension == "txt"): #plusieurs corpus a un topic chacun en entrees
            print("cas input_file_extension == txt")
            joblib.dump(LE, "./data/input/merged_corpus/labelEncoder_category_{}.joblib".format(self.topics_names_concat), 
                    compress=9)
        if(self.input_file_extension == "csv" or self.input_file_extension == "parquet"): #un seul corpus a plusieurs topic en entree
            print("cas input_file_extension == 'csv/parquet'")
            joblib.dump(LE, "./data/input/merged_corpus/labelEncoder_category_{}.joblib".format(self.merged_corpus_name), 
                    compress=9)

        # 5. Melange aleatoire des documents
        self.merged_corpus_dataframe = self.merged_corpus_dataframe.sample(frac=1).reset_index(drop=True)

        # 6. Suppression des retours a la ligne \n et \r
        self.merged_corpus_dataframe.replace("\\n", " ", regex=True, inplace=True)
        self.merged_corpus_dataframe.replace("\\r", " ", regex=True, inplace=True)

        # 7. Suppression des doublons
        print("self.merged_corpus_dataframe.shape =", self.merged_corpus_dataframe.shape)
        self.merged_corpus_dataframe.drop_duplicates("message", inplace=True, keep="first")
        print("self.merged_corpus_dataframe.shape =", self.merged_corpus_dataframe.shape)

       # 1. Creation de l'id unique
        self.merged_corpus_dataframe.index = list(range(len(self.merged_corpus_dataframe)))
        self.merged_corpus_dataframe["id"] = self.merged_corpus_dataframe.index
        # self.merged_corpus_dataframe = self.merged_corpus_dataframe[["id", "message", "message_preprocessed", "category"]]

        # Nettoyages restants a rajouter plus tard : 
        #pour enlever les faux exemples, preprocessing restant =
        #  enlever commentaires en bas d'article, description auteur, texte anglais, references bibliographiques
        #  enlever ponctuations (guillemets par exemple) 

        #Credit source : 
        #https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/


    def save_merged_corpus_dataframe(self, output_format):
        """ Enregistre le corpus dans un fichier (au format csv ou parquet) """
        if not os.path.exists("./data/input/merged_corpus/"):
            os.makedirs("./data/input/merged_corpus/")
        path = "./data/input/merged_corpus/corpus_" + self.merged_corpus_name + "." + output_format
        if(output_format == "csv"):
            self.merged_corpus_dataframe.to_csv(path, index=False, encoding="utf-8")
            corpus = pd.read_csv(path)
            print(corpus)
        elif(output_format == "parquet"):
            self.merged_corpus_dataframe.to_parquet(path, engine="fastparquet")
            corpus = pd.read_parquet(path) #engine="fastparquet"
            print(corpus)