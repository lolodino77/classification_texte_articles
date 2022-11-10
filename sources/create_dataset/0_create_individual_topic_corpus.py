import sys
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *
from lib_create_corpus import *

set_current_directory_to_root(root = "classification_texte_bapteme_philo")
print("os.getcwd() at root =", os.getcwd()) 

# rajouter la lecture des arguments entres en ligne de commande

# exemples de commande :
# python 0_create_individual_topic_corpus.py ./data/input/bibliographies/ txt
# python 0_create_individual_topic_corpus.py command parquet bibliography_middle_age_fr.txt bibliography_baptism_fr.txt
sys_argv = sys.argv
print("sys_argv =", sys_argv)
filenames = get_input_filenames(sys_argv)

for bibliography_filename in filenames:
    create_individual_topic_corpus_from_bibliographies(bibliography_filename)