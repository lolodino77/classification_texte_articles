from article import *
from bibliography import *
from blog import *
from blogspot import *
from wordpress import *
from bibliographylist import *
from bloglist import *
import os

# test2
def test_bibliographylist(filename, num_articles):
    bibliographyList = BibliographyList(filename, num_articles)
    print(bibliographyList)
    bibliographyList.save_articles_urls()
    bibliographyList.save_paragraphs()


def test_bloglist(filename, num_articles):
    blogList = BlogList(filename, num_articles)
    print(blogList)
    blogList.save_articles_urls()
    blogList.save_paragraphs()



def main():
    # args : python main.py blogs_ou_biblio.txt num_articles
    # ex : python main.py blogs_philosophy.txt 5
    # ex : python main.py bliblio_philosophy.txt 5

    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    
    sys_argv = sys.argv
    filename = sys_argv[1]
    num_articles = int(sys_argv[2])
    # filename = "blogs_philosophy.txt"
    # num_articles = 5
    
    # test_bloglist(filename, num_articles)
    test_bibliographylist(filename, num_articles)

main()