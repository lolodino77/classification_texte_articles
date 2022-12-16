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
    """ Classe pour fusionner plusieurs corpus en un seul corpus
    1. Soit un corpus final avec un seul et unique topic
    2. Soit un corpus final avec plusieurs topics
    """

    def __init__(self, corpus_txt_list_filename, language):
        self.language = language
        self.corpus_txt_list_filename = corpus_txt_list_filename
        self.corpus_txt_filenames, self.topics = self.create_corpus_txt_filenames_and_topics() 
        self.topics_names_concat = "_".join(sorted(set(self.topics)))
        self.corpus_dataframes = self.create_corpus_dataframes()
        self.merged_corpus_name = self.create_merged_corpus_name()
        self.merged_corpus_dataframe = self.create_merged_corpus_dataframe()


    def __str__(self):
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


    def create_merged_corpus_name(self):
        corpus_txt_names = []
        for filename in self.corpus_txt_filenames:
            corpus_txt_names.append(get_corpus_name_from_filename(filename))
        merged_corpus_name = "_".join(corpus_txt_names)
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
        """ Renvoie le corpus pandas dataframe a partir d'un corpus au format .txt"""
        res = open("./data/input/corpus_txt/" + corpus_txt_filename, "r", encoding="utf-8").read().split("\n\n")
        res = [elt for elt in res if len(elt) > 1]
        message = res
        length = [len(elt) for elt in res]
        list_of_rows = list(zip(message, length))
        corpus_dataframe = pd.DataFrame(list_of_rows, columns=["message", "length"])
        corpus_dataframe["category"] = topic

        return(corpus_dataframe)


    def create_corpus_dataframes(self):
        """ Renvoie une liste de dataframes (chaque corpus) a partir des fichiers .txt """
        all_corpus_dataframes = []
        for filename, topic in zip(self.corpus_txt_filenames, self.topics):
            corpus_dataframe = self.get_corpus_dataframe(filename, topic)
            all_corpus_dataframes.append(corpus_dataframe)
        
        return(all_corpus_dataframes)


    def create_merged_corpus_dataframe(self):
        """ Renvoie le dataframe final qui est la fusion de tous les dataframes """
        if(len(self.corpus_dataframes) > 1):
            merged_corpus = pd.concat(self.corpus_dataframes) 
        else:
            merged_corpus = self.corpus_dataframes[0]
        
        return(merged_corpus)
    

    def get_preprocessed_messages(self, list_of_documents, lemmatizer, stopwords):
        """Nettoie tous les documents d'une liste pour creer un dataset exploitable par des modeles d'IA.
        
        Parametres:
        list_of_documents (liste de string) : Une liste de documents (les textes a classifier) a nettoyer 
        lemmatizer (fonction) : Le lemmatizer qui servira a lemmatizer les mots des documents si possible
        stopwords (liste de string) : La liste des stopwords (mots frequents mais inutiles a enlever)

        Sortie:
        preprocess_list (liste de string) : Une liste de documents nettoyes
        """
    # cas speciaux restants a traiter :
    # mots avec un apostrophe avant (Traite)
    # mots composes avec un ou plusieurs tirets (A traiter)
        preprocess_list = []
        for document in list_of_documents :
            #remplacer les virgules bizarres
            document = document.replace("’", "'")

            # supprimer les mots avant les apostrophes (particules comme l', t', etc.)
            document = re.sub(r"\s\w+'", " ", document, 0)

            # enlever la ponctuation et met en minuscule
            ponctuation_to_remove = string.punctuation.replace("-", "")
            document_w_punct = "".join([i.lower() for i in document if i not in ponctuation_to_remove])

            # enlever les chiffres
            document_w_num = ''.join(i for i in document_w_punct if not i.isdigit())

            # transformer les phrases en liste de tokens (en liste de mots)
            tokenize_document = nltk.tokenize.word_tokenize(document_w_num)

            # enlever les stopwords (mots n’apportant pas de sens)
            words_w_stopwords = [i for i in tokenize_document if i not in stopwords]

            # lemmatizer (convertir en la racine)
            words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords) #words_lemmatize est un iterateur
            words_lemmatize = list(words_lemmatize)

            # reformer la phrase en reliant les mots precedents
            document_clean = " ".join(words_lemmatize)

            #rajouter la phrase dans la liste
            preprocess_list.append(document_clean)
            
        return preprocess_list 


    def preprocess_merged_corpus_dataframe(self):
        # Recuperation du lemmatizer
        stopwords = nltk.corpus.stopwords.words(self.language)
        print("os.getcwd() =", os.getcwd())
        # mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))
        if(self.language == "french"):
            lemmatizer = FrenchLefffLemmatizer()
        elif(self.language == "english"):
            lemmatizer = WordNetLemmatizer() #le lemmatizer WordNetLemmatizer de nltk uniquement pour l'anglais 

        # Execution de la fonction principale qui fait le nettoyage
        self.merged_corpus_dataframe["message_preprocessed"] = self.get_preprocessed_messages(self.merged_corpus_dataframe['message'], 
                                                                        lemmatizer, stopwords)

        # Creation de l'id unique
        self.merged_corpus_dataframe.index = list(range(len(self.merged_corpus_dataframe)))
        self.merged_corpus_dataframe["id"] = self.merged_corpus_dataframe.index

        # Suppression des colonnes inutiles
        self.merged_corpus_dataframe = self.merged_corpus_dataframe[["id", "message", "message_preprocessed", "category"]]

        # Creation de la taille de chaque documents (en nombre de caracteres)
        self.merged_corpus_dataframe["length"] = self.merged_corpus_dataframe["message"].str.len()

        # Annotation au format entier (necessaire pour certaines fonctions de sklearn)
        # self.merged_corpus_dataframe["category_bin"] = np.select([self.merged_corpus_dataframe["category"] == class_1], [1], default=0)
        LE = LabelEncoder()
        self.merged_corpus_dataframe["category_bin"] = LE.fit_transform(self.merged_corpus_dataframe["category"]) 
        joblib.dump(LE, "./data/input/merged_corpus/labelEncoder_category_{}.joblib".format(self.topics_names_concat), 
                    compress=9)

        # Melange aleatoire des documents
        self.merged_corpus_dataframe = self.merged_corpus_dataframe.sample(frac=1).reset_index(drop=True)

        # Suppression des retours a la ligne \n et \r
        self.merged_corpus_dataframe.replace("\\n", " ", regex=True, inplace=True)
        self.merged_corpus_dataframe.replace("\\r", " ", regex=True, inplace=True)

        # Suppression des doublons
        print("self.merged_corpus_dataframe.shape =", self.merged_corpus_dataframe.shape)
        self.merged_corpus_dataframe.drop_duplicates("message", inplace=True, keep="first")
        print("self.merged_corpus_dataframe.shape =", self.merged_corpus_dataframe.shape)

        #pour enlever les faux exemples, preprocessing restant =
        #  enlever commentaires en bas d'article, description auteur, texte anglais, references bibliographiques
        #  enlever ponctuations (guillemets par exemple) 

        #Credit source : 
        #https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/

        print("message_preprocessed =", self.merged_corpus_dataframe["message_preprocessed"])


    def save_merged_corpus_dataframe(self, output_format):
        # Enregistrer le corpus (au format csv ou parquet)
        if not os.path.exists("./data/input/merged_corpus/"):
            os.makedirs("./data/input/merged_corpus/")
        path = "./data/input/merged_corpus/corpus_" + self.topics_names_concat + "." + output_format
        if(output_format == "csv"):
            self.merged_corpus_dataframe.to_csv(path, index=False, encoding="utf-8")
            corpus = pd.read_csv(path)
            print(corpus)
        elif(output_format == "parquet"):
            self.merged_corpus_dataframe.to_parquet(path, engine="fastparquet")
            corpus = pd.read_parquet(path) #engine="fastparquet"
            print(corpus)