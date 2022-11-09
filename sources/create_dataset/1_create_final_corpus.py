import sys
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *
from lib_corpus_creation import *

set_current_directory_to_root(root = "classification_texte_bapteme_philo")
print("os.getcwd() at root =", os.getcwd()) 

# exemple de commande :
# python create_final_corpus.py dataset_baptism_fr.csv dataset_philosophy_fr.csv
filename_corpus_topic_1 = sys.argv[1]
filename_corpus_topic_2 = sys.argv[2]

# corpus_datasets_names = ["dataset_baptism_fr.csv", "dataset_philosophy_fr.csv"]
topic_1 = get_topic_from_filename(filename_corpus_topic_1, keep_language=False)
topic_2 = get_topic_from_filename(filename_corpus_topic_2, keep_language=False)
topics = [topic_1, topic_2]
file_extension = "parquet"
corpus_datasets_names = [filename_corpus_topic_1, filename_corpus_topic_2]
final_corpus_name = "data_{}_{}.".format(topic_1, topic_2) + file_extension
language = "french"

write_multiple_topics_corpus_dataset(corpus_datasets_names, final_corpus_name, topics, language)