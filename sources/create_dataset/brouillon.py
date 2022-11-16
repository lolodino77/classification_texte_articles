# from lib_create_articles_lists import *
import sys
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *
from lib_create_corpus import *

set_current_directory_to_root(root = "classification_texte_bapteme_philo")
print("os.getcwd() at root =", os.getcwd())

path_to_directory = "./data/input/corpus_txt/"
# print("path_to_directory =", path_to_directory)
# L = get_all_files_from_a_directory(path_to_directory, files_extension="")
# print(L)

file_list_of_blogs = "blogs_philosophy_eng.txt"
table_extension = "csv"
write_all_corpus_txt_to_corpus_table(file_list_of_blogs, table_extension)