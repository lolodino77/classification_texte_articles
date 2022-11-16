import sys
from pathlib import Path, PureWindowsPath
sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources").as_posix())
from lib_general import *
from lib_create_corpus import *
from lib_create_corpus_method_bibliographies import *
# from lib_create_articles_lists_method_bibliographies import *

set_current_directory_to_root(root = "classification_texte_bapteme_philo")
print("os.getcwd() at root =", os.getcwd()) 

# rajouter la lecture des arguments entres en ligne de commande

# pattern de commande
# all_articles, num_articles

# exemples de commande :
# exemple 1 garder tous les articles de chaque bibliographie
# python 0_create_corpus_from_bibliographies.py ./data/input/bibliographies/ txt

# exemple 2 avec 8 comme nombre d'articles par bibliographie :
# python 0_create_corpus_from_bibliographies.py 8 ./data/input/bibliographies/ txt

# exemple 3 utiliser des bibliographies precises appelees par leur nom
# python 0_create_corpus_from_bibliographies.py all command parquet bibliography_middle_age_fr.txt bibliography_baptism_fr.txt

# exemple 4 utiliser des bibliographies precises appelees par leur nom et seulement 8 articles par biblio
# python 0_create_corpus_from_bibliographies.py 8 command parquet bibliography_middle_age_fr.txt bibliography_baptism_fr.txt


sys_argv = sys.argv
print("sys_argv =", sys_argv)
bibliographies_filenames = get_bibliographies_filenames(sys_argv)
all_articles = get_var_all_articles(sys_argv)
num_articles = get_var_num_articles(sys_argv)
print("all_articles =", all_articles)
print("num_articles =", num_articles)
print("type all_articles =", type(all_articles))
print("type num_articles =", type(num_articles))

print("\n\n\nbibliographies_filenames =", bibliographies_filenames)
save_multiple_corpus_from_bibliographies_lists_files(bibliographies_filenames, all_articles, num_articles)