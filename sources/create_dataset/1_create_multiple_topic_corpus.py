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


def run_corpusMerger(filename, output_format, language):
    corpusMerger = CorpusMerger(filename, language)
    corpusMerger.preprocess_merged_corpus_dataframe()
    corpusMerger.save_merged_corpus_dataframe(output_format)


def test_corpusMerger_input_two_topics():
    filename = "corpus_amazon_raw.parquet"
    language = "english"
    output_format = "csv"

    cm = CorpusMerger(filename, language)
    cm.preprocess_merged_corpus_dataframe()
    cm.save_merged_corpus_dataframe(output_format)

    return(cm.merged_corpus_dataframe)


def test_corpusMerger_input_one_topic():
    filename = "list_corpus_philosophy.txt"
    language = "english"
    output_format = "csv"

    cm = CorpusMerger(filename, language)
    cm.preprocess_merged_corpus_dataframe()
    cm.save_merged_corpus_dataframe(output_format)

    return(cm.merged_corpus_dataframe)


def main():
    # args : python create_multiple_topic_corpus.py corpus_lists.txt
    ### corpus_lists.txt : un fichier .txt qui contient chaque corpus a fusionner et son topic

    # ex : 
    # python 1_create_multiple_topic_corpus.py list_corpus_philosophy.txt english csv
    # python 1_create_multiple_topic_corpus.py list_corpus_philosophy.txt english parquet
    # python 1_create_multiple_topic_corpus.py list_corpus_feser_pruss.txt english csv
    # python 1_create_multiple_topic_corpus.py corpus_amazon.parquet english parquet

    set_current_directory_to_root(root = "classification_texte_articles")
    print("os.getcwd()")
    print(os.getcwd())
    
    sys_argv = sys.argv
    print("in 1_create_multiple_topic_corpus.py")
    print("sys_argv =", sys_argv)
    filename = sys_argv[1]
    language = sys_argv[2]
    output_format = sys_argv[3]

    run_corpusMerger(filename, output_format, language)
    

main()