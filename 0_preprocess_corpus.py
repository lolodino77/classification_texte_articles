import pandas as pd
import numpy as np
import nltk
import re
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
pd.set_option('display.max_colwidth', 600)
pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

df = pd.read_csv('dataset_philosophy.csv')
# print(df)

#Pas besoin si tout est deja installe
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('wordnet')

language = "french"
french_stopwords = nltk.corpus.stopwords.words(language)
mots = set(line.strip() for line in open('dictionnaire.txt'))
lemmatizer = FrenchLefffLemmatizer()

def preprocess_list_of_documents(listofdocuments):
# cas speciaux a traiter
# mots avec un apostrophe avant (Traite)
# mots composes avec un ou plusieurs tirets (A traiter)
	preprocess_list = []
	for document in listofdocuments :
		#remplacer les virgules bizarres
		document = document.replace("’", "'")

		# supprimer les mots avant les apostrophes (particules comme l', t', etc.)
		document = re.sub(r"\s\w+'", " ", document, 0)

		# enlever la ponctuation et met en minuscule
		document_w_punct = "".join([i.lower() for i in document if i not in string.punctuation])

		# enlever les chiffres
		document_w_num = ''.join(i for i in document_w_punct if not i.isdigit())

		# transformer les phrases en liste de tokens (en liste de mots)
		tokenize_document = nltk.tokenize.word_tokenize(document_w_num)

		# enlever les stopwords (mots n’apportant pas de sens)
		words_w_stopwords = [i for i in tokenize_document if i not in french_stopwords]

		# lemmatizer (convertir en la racine)
		words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords) #words_lemmatize est un iterateur
		words_lemmatize = list(words_lemmatize)

		# reformer la phrase en reliant les mots precedents
		document_clean = " ".join(words_lemmatize)

		#rajouter la phrase dans la liste
		preprocess_list.append(document_clean)
	return preprocess_list


preprocessed_corpus = df.copy()
preprocessed_corpus['message'] = preprocess_list_of_documents(df['message'])
print("preprocessed_corpus :")
print(preprocessed_corpus)
print(type(preprocessed_corpus))

# enregistre le corpus nettoye
filename = "dataset_philosophy_preprocessed.csv"
preprocessed_corpus.to_csv(filename, index=False, sep=",")




#Credit source : 
#https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/