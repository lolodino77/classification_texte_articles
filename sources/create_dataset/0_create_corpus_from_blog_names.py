# rajouter la lecture des arguments entres en ligne de commande

# exemples de commande :
# python 0_create_individual_topic_corpus_from_bibliographies.py ./data/input/bibliographies/ txt
# python 0_create_individual_topic_corpus_from_bibliographies.py command parquet bibliography_middle_age_fr.txt bibliography_baptism_fr.txt

from lib_create_articles_lists import *
from lib_create_articles_lists_method_blogs import *
from lib_create_corpus import *
from lib_create_corpus_method_blogs import *
import sys
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *

set_current_directory_to_root(root = "classification_texte_bapteme_philo")
# print("os.getcwd() at root =", os.getcwd())

# exemples de commande :
# python 0_create_corpus_from_blog_names.py file_list_of_blogs input_file_extension output_file_extension
# python 0_create_corpus_from_blog_names.py blogs_philosophy_eng.txt txt csv
# python 0_create_corpus_from_blog_names.py blogs_philosophy_eng.txt txt parquet
sys_argv = sys.argv
file_list_of_blogs = sys_argv[1]
input_file_extension = sys_argv[2]
output_file_extension = sys_argv[3]

file_list_of_blogs = "blogs_philosophy_eng.txt"
input_file_extension = "txt"
output_file_extension = "csv"
create_corpus_csv_from_blogs_names(file_list_of_blogs, input_file_extension, output_file_extension)