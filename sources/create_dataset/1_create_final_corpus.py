import sys
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *
# from  import *
from lib_create_articles_lists import *
from lib_merge_corpus import *
import pandas as pd 

set_current_directory_to_root(root = "classification_texte_bapteme_philo")
print("os.getcwd() at root =", os.getcwd()) 

# modele de commande
# python create_final_corpus.py filename_corpus_0 filename_corpus_1 output_extension language

# exemple de commande :
# python 1_create_final_corpus.py corpus_baptism_fr.csv corpus_philosophy_fr.csv csv french
# python 1_create_final_corpus.py corpus_baptism_fr.csv corpus_philosophy_fr.csv csv english
# python 1_create_final_corpus.py corpus_edwardfeser.parquet corpus_alexanderpruss.parquet parquet english

filename_corpus_topic_1 = sys.argv[1]
filename_corpus_topic_2 = sys.argv[2]
output_extension = sys.argv[3]
# output_extension = "parquet"
language = sys.argv[4]
# language = "french"


# corpus_datasets_names = ["dataset_baptism_fr.csv", "dataset_philosophy_fr.csv"]
topic_1 = get_topic_from_filename(filename_corpus_topic_1, keep_language=False)
topic_2 = get_topic_from_filename(filename_corpus_topic_2, keep_language=False)
topics = [topic_1, topic_2]
corpus_filenames = [filename_corpus_topic_1, filename_corpus_topic_2]
print("corpus_filenames =", corpus_filenames)
print("topics =", topics)
final_corpus_name = "corpus_{}_{}.".format(topic_1, topic_2) + output_extension

merged_corpus = merge_two_corpus(corpus_filenames, final_corpus_name, topics, language)
save_merged_corpus_table(merged_corpus, topic_1, topic_2, output_extension)
print(merged_corpus["category"].value_counts())