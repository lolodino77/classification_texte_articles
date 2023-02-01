from article import *
from bibliography import *
from blog import *
from blogspot import *
from wordpress import *
from bibliographylist import *
from bloglist import *
import os
from corpusmerger import *

import sys
sys.path.append("../")


def test_corpusMerger(corpus_txt_list_filename, output_format, language):
    corpusMerger = CorpusMerger(corpus_txt_list_filename, language)
    corpusMerger.preprocess_merged_corpus_dataframe()
    corpusMerger.save_merged_corpus_dataframe(output_format)


def main():
    # args : python create_multiple_topic_corpus.py corpus_lists.txt
    ### corpus_lists.txt : un fichier .txt qui contient chaque corpus a fusionner et son topic

    # ex : 
    # python 1_create_multiple_topic_corpus.py corpus_list_philosophy.txt english csv
    # python 1_create_multiple_topic_corpus.py corpus_list_philosophy.txt english parquet
    # python 1_create_multiple_topic_corpus.py corpus_list_philosophie.txt french csv

    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    
    sys_argv = sys.argv
    print("in 1_create_multiple_topic_corpus.py")
    print("sys_argv =", sys_argv)
    corpus_txt_list_filename = sys_argv[1]
    language = sys_argv[2]
    output_format = sys_argv[3]

    test_corpusMerger(corpus_txt_list_filename, output_format, language)
    

main()