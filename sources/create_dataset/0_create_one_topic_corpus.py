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


# from2
def from_bibliographylist(filename, num_articles, table_format):
    print("in from_bibliographylist()")
    bibliographyList = BibliographyList(filename, num_articles, table_format)
    print(bibliographyList)
    bibliographyList.save_articles_urls()
    bibliographyList.save_corpus_txt()
    bibliographyList.save_corpus_dataframe()


def from_bloglist(filename, num_articles, table_format):
    print("in from_bloglist")
    blogList = BlogList(filename, num_articles, table_format)
    print("prout")
    print(blogList)
    blogList.save_articles_urls()
    blogList.save_corpus_txt()
    blogList.save_corpus_dataframe()


def create_one_topic_corpus(filename, num_articles, table_format):
    if("bibliography" in filename):
        from_bibliographylist(filename, num_articles, table_format)
    elif("blog" in filename):
        from_bloglist(filename, num_articles, table_format)


def main():
    # args : python 0_create_one_topic_corpus.py blogs_ou_biblio.txt num_articles
    # ex : 
    # python 0_create_one_topic_corpus.py blogs_philosophy.txt 5 csv
    # python 0_create_one_topic_corpus.py blogs_philosophy.txt 5 parquet
    # python 0_create_one_topic_corpus.py blogs_all_articles.txt all csv
    # python 0_create_one_topic_corpus.py bibliography_philosophy.txt 5 csv
    # python 0_create_one_topic_corpus.py bibliography_philosophy.txt 5 parquet

    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    
    sys_argv = sys.argv
    filename = sys_argv[1]
    num_articles = sys_argv[2]
    table_format = sys_argv[3]
    print("in 0_create_one_topic_corpus.py")
    print("sys_argv =", sys_argv)

    print("filename = ", filename)
    print("num_articles = ", num_articles)
    print("table_format = ", table_format)

    if(num_articles != "all"):
        num_articles = int(num_articles)

    create_one_topic_corpus(filename, num_articles, table_format)


main()