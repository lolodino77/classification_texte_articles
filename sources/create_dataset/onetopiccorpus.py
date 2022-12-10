from lib_general import *
import nltk
import re
import string
import pandas as pd


class OneTopicCorpus:
    """ Definit un corpus qui contient des documents d'un seul topic """

    def __init__(self, filename_input, topic):
        """ Constructor of Article 
        filename_input (string) : nom du fichier qui contient les noms des fichiers .txt des corpus a fusionner
        corpus_txt_filenames (liste de string) : liste des fichiers .txt des corpus a fusionner
        corpus_dataframes (liste de pandas dataframes) : liste de pandas dataframes 
        """
        self.topic = topic
        self.filename_input = filename_input
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


    def get_corpus_dataframe(self, filename_corpus_txt):
        """ Renvoie le corpus pandas dataframe a partir d'un corpus au format .txt"""
        res = open("./data/input/corpus_txt/" + filename_corpus_txt, "r", encoding="utf-8").read().split("\n\n")
        res = [elt for elt in res if len(elt) > 1]
        message = res
        length = [len(elt) for elt in res]
        list_of_rows = list(zip(message, length))
        corpus_dataframe = pd.DataFrame(list_of_rows, columns=["message", "length"])

        return(corpus_dataframe)


    def create_corpus_txt_filenames(self):
        corpus_txt_filenames = open("./data/input/corpus_txt/" + self.filename_input, "r", encoding="utf-8").read().split("\n")
        return(corpus_txt_filenames)


    def create_corpus_dataframes(self):
        """ Renvoie une liste de dataframes (chaque corpus) a partir des fichiers .txt """
        all_corpus_dataframes = []
        for filename in self.corpus_txt_filenames:
            corpus_dataframe = self.get_corpus_dataframe(filename)
            all_corpus_dataframes.append(corpus_dataframe)
        
        return(all_corpus_dataframes)


    def create_corpus_merged_dataframe(self):
        """ Renvoie le dataframe final qui est la fusion de tous les dataframes """
        if(len(self.corpus_dataframes) > 1):
            corpus_merged = pd.concat(self.corpus_dataframes) 
        else:
            corpus_merged = self.corpus_dataframes[0]
        corpus_merged["category"] = self.topic
        
        return(corpus_merged)


    def preprocess_list_of_documents(list_of_documents, lemmatizer, stopwords):
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


    def merge_two_corpus(corpus_filenames, final_corpus_name, topics, language):
        """Cree un corpus d'un topic au format pandas dataframe dans le fichier texte filename_output
        Marche pour l'instant que pour fusionner deux corpus (deux topics differents)
        To do : faire pour classification multiclasse

        Parametres: 
        corpus_filenames (liste de string) : Les noms des datasets de corpus a fusionner
                        Exemple : ["corpus_philosophy_fr.txt", "corpus_history_fr.txt", "corpus_animals_fr.txt"]
        final_corpus_name (string) : Le nom du fichier dans lequel on ecrira le corpus sous format csv
                        Exemple : "dataset_philosophy_history_fr.txt", "dataset_philosophy_history_animals_fr.txt"
        topics (liste de string) : Le nom des topics
                        Exemple : ["philosophy", "history"]
        language (string) : La langue des documents
                        Valeurs possibles : "french" ou "english"
        
        Sortie:
        None : Fichier filename_corpus_output qui contient le corpus au format pandas dataframe
        """
        #Pas besoin si tout est deja installe
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('words')
        nltk.download('wordnet')

        #Lecture des fichiers csv
        # print(os.getcwd())
        # os.chdir(os.path.dirname(os.path.abspath(__file__ + '/..' * 2)))
        # print(os.getcwd())

        # creer version soit csv soit parquet
        corpus_0 = get_corpus_table_from_filename(corpus_0_filename)
        corpus_1 = get_corpus_table_from_filename(corpus_1_filename)
        
        # Annotation des documents
        class_0 = topics[0]
        class_1 = topics[1]
        corpus_0["category"] = class_0
        corpus_1["category"] = class_1

        # Creation du dataset final en regroupant les documents des deux classes
        merged_corpus = pd.concat([corpus_1, corpus_0]) 

        # Recuperation du lemmatizer
        stopwords = nltk.corpus.stopwords.words(language)
        print("os.getcwd() =", os.getcwd())
        # mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))
        if(language == "french"):
            lemmatizer = FrenchLefffLemmatizer()
        elif(language == "english"):
            lemmatizer = WordNetLemmatizer() #le lemmatizer WordNetLemmatizer de nltk uniquement pour l'anglais 

        # Execution de la fonction principale qui fait le nettoyage
        merged_corpus["message_preprocessed"] = preprocess_list_of_documents(merged_corpus['message'], lemmatizer, stopwords)

        # Creation de l'id unique
        merged_corpus.index = list(range(len(merged_corpus)))
        merged_corpus["id"] = merged_corpus.index

        # Suppression des colonnes inutiles
        merged_corpus = merged_corpus[["id", "message", "message_preprocessed", "category"]]

        # Creation de la taille de chaque documents (en nombre de caracteres)
        merged_corpus["length"] = merged_corpus["message"].str.len()

        # Annotation au format entier (necessaire pour certaines fonctions de sklearn)
        merged_corpus["category_bin"] = np.select([merged_corpus["category"] == class_1], [1], default=0)

        # Melange aleatoire des documents
        merged_corpus = merged_corpus.sample(frac=1).reset_index(drop=True)

        # Suppression des retours a la ligne \n et \r
        merged_corpus.replace("\\n", " ", regex=True, inplace=True)
        merged_corpus.replace("\\r", " ", regex=True, inplace=True)

        # Suppression des doublons
        print("merged_corpus.shape =", merged_corpus.shape)
        merged_corpus.drop_duplicates("message", inplace=True, keep="first")
        print("merged_corpus.shape =", merged_corpus.shape)

        #pour enlever les faux exemples, preprocessing restant =
        #  enlever commentaires en bas d'article, description auteur, texte anglais, references bibliographiques
        #  enlever ponctuations (guillemets par exemple) 

        #Credit source : 
        #https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/

        return(merged_corpus)