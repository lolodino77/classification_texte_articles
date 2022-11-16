import nltk
from nltk.stem import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer



def merge_two_corpus(corpus_datasets_names, final_corpus_name, topics, language):
	"""Cree un corpus d'un topic au format pandas dataframe dans le fichier texte filename_output
	Marche pour l'instant que pour fusionner deux corpus (deux topics differents)
	To do : faire pour classification multiclasse

	Parametres: 
	corpus_datasets_names (liste de string) : Les noms des datasets de corpus a fusionner
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
	corpus_0 = pd.read_csv('data/input/' + corpus_datasets_names[0])
	corpus_1 = pd.read_csv('data/input/' + corpus_datasets_names[1])
	
	# Annotation des documents
	class_0 = topics[0]
	class_1 = topics[1]
	corpus_0["category"] = class_0
	corpus_1["category"] = class_1

	# Creation du dataset final en regroupant les documents des deux classes
	corpus = pd.concat([corpus_1, corpus_0]) 

	# Recuperation du lemmatizer
	stopwords = nltk.corpus.stopwords.words(language)
	print("os.getcwd() =", os.getcwd())
	# mots = set(line.strip() for line in open('dictionnaire.txt', encoding="utf8"))
	if(language == "french"):
		lemmatizer = FrenchLefffLemmatizer()
	elif(language == "english"):
		lemmatizer = WordNetLemmatizer() #le lemmatizer WordNetLemmatizer de nltk uniquement pour l'anglais 

	# Execution de la fonction principale qui fait le nettoyage
	corpus["message_preprocessed"] = preprocess_list_of_documents(corpus['message'], lemmatizer, stopwords)

	# Creation de l'id unique
	corpus.index = list(range(len(corpus)))
	corpus["id"] = corpus.index

	# Suppression des colonnes inutiles
	corpus = corpus[["id", "message", "message_preprocessed", "category"]]

	# Creation de la taille de chaque documents (en nombre de caracteres)
	corpus["length"] = corpus["message"].str.len()

	# Annotation au format entier (necessaire pour certaines fonctions de sklearn)
	corpus["category_bin"] = np.select([corpus["category"] == class_1], [1], default=0)

	# Melange aleatoire des documents
	corpus = corpus.sample(frac=1).reset_index(drop=True)

	# Suppression des retours a la ligne \n et \r
	corpus.replace("\\n", " ", regex=True, inplace=True)
	corpus.replace("\\r", " ", regex=True, inplace=True)

	# Suppression des doublons
	print("corpus.shape =", corpus.shape)
	corpus.drop_duplicates("message", inplace=True, keep="first")
	print("corpus.shape =", corpus.shape)

	#pour enlever les faux exemples, preprocessing restant =
	#  enlever commentaires, description auteur, texte anglais, references bibliographiques
	#  enlever ponctuations (guillemets par exemple) 

	#Credit source : 
	#https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/

	return(corpus)


def save_merged_corpus_table(corpus, class_0, class_1, table_extension):
	# Enregistrer le corpus (au format parquet)
	path = "./data/input/data_" + class_1 + "_" + class_0 + ".parquet"
	corpus.to_parquet(path, engine="fastparquet")
	corpus = pd.read_parquet(path) #engine="fastparquet"
	print(corpus)