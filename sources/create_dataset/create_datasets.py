import sys
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *
from lib_corpus_creation import *

set_current_directory_to_root(root = "classification_texte_bapteme_philo")
print("os.getcwd() at root =", os.getcwd()) 

# rajouter la lecture des arguments entres en ligne de commande

# exemples de commande :
# python create_datasets.py in_input_repertory txt
# python create_datasets.py in_command parquet bibliography_middle_age_fr.txt bibliography_baptism_fr.txt
sys_argv = sys.argv
filenames = get_input_filenames(sys_argv)

# filenames = ["bibliography_middle_age_fr.txt", "bibliography_baptism_fr.txt"]
# filenames = ["bibliography_philosophy_fr.txt", "bibliography_baptism_fr.txt", "bibliography_middle_age_fr.txt"]
for filename in filenames:
    f = open(os.getcwd() + "\\data\\input\\bibliographies\\" + filename, "r")
    bibliography_urls = f.read().split("\n")
    filename = filename.split(".txt")[0]
    topic = filename.split("_")[1:] 
    topic = "_".join(topic)
    filename_urls_articles = "urls_{}_articles.txt".format(topic)
    filename_corpus = "corpus_{}.txt".format(topic)
    filename_corpus_input = "corpus_{}.txt".format(topic)
    filename_corpus_output = "dataset_{}.csv".format(topic)

    write_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode="a")
    write_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode="a")

