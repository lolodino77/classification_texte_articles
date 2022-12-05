from lib_general import *
import nltk
import re
import string


class Corpus:
    """ Define an article from a blog """

    def __init__(self, corpus_filenames):
        """ Constructor of Article """
        self.corpus_filenames = corpus_filenames
        self.paragraphs = self.create_paragraphs()


    def __str__(self):
        """ Renvoie une chaine de caractère décrivant l'article """
        print("str :")
        # str_url = str(self.url)
        # # str_articles_list = str(self.articles_list)
        # str_paragraphs = str(self.paragraphs)
        # desc = "url = "+ str_url
        # desc += "\nparagraphs = " + str_paragraphs 
        # # desc = desc + "\nfilename = " + str_filename
        # return(desc)  


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