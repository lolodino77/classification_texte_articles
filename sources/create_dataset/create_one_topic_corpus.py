from article import *
from bibliography import *
from blog import *
from blogspot import *
from wordpress import *
from bibliographylist import *
from bloglist import *
import os

import sys
sys.path.append("../")


# test2
def test_bibliographylist(filename, num_articles, table_format):
    bibliographyList = BibliographyList(filename, num_articles, table_format)
    print(bibliographyList)
    bibliographyList.save_articles_urls()
    bibliographyList.save_corpus_txt()
    bibliographyList.save_corpus_dataframe()


def test_bloglist(filename, num_articles, table_format):
    blogList = BlogList(filename, num_articles, table_format)
    print(blogList)
    blogList.save_articles_urls()
    blogList.save_corpus_txt()
    blogList.save_corpus_dataframe()


def create_one_topic_corpus(filename, num_articles, table_format):
    if("bibliography" in filename):
        test_bibliographylist(filename, num_articles, table_format)
    elif("blog" in filename):
        test_bloglist(filename, num_articles, table_format)


def main():
    # args : python create_one_topic_corpus.py blogs_ou_biblio.txt num_articles
    # ex : 
    # python create_one_topic_corpus.py blogs_philosophy.txt 5 csv
    # python create_one_topic_corpus.py blogs_philosophy.txt 5 parquet
    # python create_one_topic_corpus.py blogs_all_articles.txt all csv
    # python create_one_topic_corpus.py bibliography_philosophy.txt 5 csv
    # python create_one_topic_corpus.py bibliography_philosophy.txt 5 parquet

    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    
    sys_argv = sys.argv
    filename = sys_argv[1]
    num_articles = sys_argv[2]
    table_format = sys_argv[3]

    if(num_articles != "all"):
        num_articles = int(num_articles)

    create_one_topic_corpus(filename, num_articles, table_format)


main()