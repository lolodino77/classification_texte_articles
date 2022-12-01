from article import *
from bibliography import *
from blog import *
from blogspot import *
from bibliographylist import *
from bloglist import *
import os


def test_article(url, corpus_name):
    article = Article(url, corpus_name)
    print(article)
    path_corpus = "./data/input/corpus_txt/corpus_{}.txt".format(corpus_name)  #self.filename
    article.save_paragraphs(path_corpus, corpus_paragraphs="", file_open_mode="w", sep = "\n\n")
    print("article.save_paragraphs() fini")


def test_datasource(url, corpus_name, num_articles):
    datasource = DataSource(url, corpus_name, num_articles)
    print(datasource)


def test_bibliography(url, corpus_name, num_articles):
    bibliography = Bibliography(url, corpus_name, num_articles)
    print(bibliography)
    bibliography.save_paragraphs(savemode="overwrite")
    # print("bibliography.save_paragraphs() fini")


def test_bibliographylist(filename, num_articles):
    bibliographyList = BibliographyList(filename, num_articles)
    print(bibliographyList)
    bibliographyList.save_articles_urls()
    bibliographyList.save_paragraphs()


def test_blog(url, num_articles):
    blog = Blog(url, num_articles)
    print(blog)
    # blog.save_articles_urls()
    # blog.save_paragraphs(savemode="overwrite")


def test_bloglist(filename, num_articles):
    blogList = BlogList(filename, num_articles)
    print(blogList)
    blogList.save_articles_urls()
    blogList.save_paragraphs()


def test_blogspot(url, num_articles):
    blogspot = Blogspot(url, num_articles)
    print(blogspot)
    blogspot.save_articles_urls()
    blogspot.save_paragraphs()


def main():
    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    # url = "https://parlafoi.fr/lire/series/le-pedobapteme/"
    url = "https://parlafoi.fr/lire/series/la-scolastique-protestante/"
    url2 = "https://parlafoi.fr/lire/series/la-scolastique-protestante/"
    url3 = "https://parlafoi.fr/lire/series/commentaire-de-la-summa/"
    corpus_name = "moyen_age"
    num_articles = 40
    # num_articles = "all"

    # test_article(url, corpus_name)
    # test_datasource(url, corpus_name, num_articles)
    # test_bibliography(urls_biblio, corpus_name, num_articles)
    # test_blog(url_blog, num_articles)

    # filename = "bibliography_middle_age_fr.txt"
    # num_articles = 6
    # test_bibliographylist(filename, num_articles)

    filename = "blogs_philosophy_eng.txt"
    num_articles = 5
    test_bloglist(filename, num_articles)

    # url = "https://edwardfeser.blogspot.com"
    # num_articles = 3
    # test_blogspot(url, num_articles)

main()