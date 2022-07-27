import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer
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
stopwords = nltk.corpus.stopwords.words(language)
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()

def Preprocess_listofSentence(listofSentence):
	preprocess_list = []
	for sentence in listofSentence :
		# enlever la ponctuation et met en minuscule
		sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])

		# enlever les chiffres
		sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())

		# transformer les phrases en liste de tokens (en liste de mots)
		tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)

		# enlever les stopwords (mots n’apportant pas de sens)
		words_w_stopwords = [i for i in tokenize_sentence if i not in stopwords]

		# lemmatizer
		words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)

		# enlever les majuscules
		sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in words or not w.isalpha())

		# reformer les phrases avec les mots restants
		preprocess_list.append(sentence_clean)
	return preprocess_list


preprocessed_corpus = df.copy()
preprocessed_corpus['message'] = Preprocess_listofSentence(df['message'])
print("preprocessed_corpus :")
print(preprocessed_corpus)
print(type(preprocessed_corpus))

# enregistre le corpus nettoye
filename = "dataset_philosophy_preprocessed.csv"
preprocessed_corpus.to_csv(filename, index=False)




#Credit source : https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/