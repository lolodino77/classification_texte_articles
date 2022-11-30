from article import *
from bibliography import *
from blog import *
from bibliographylist import *
from bloglist import *
import os


def test_article(url, topic):
    article = Article(url, topic)
    print(article)
    path_corpus = "./data/input/corpus_txt/corpus_{}.txt".format(topic)  #self.filename
    article.save_paragraphs(path_corpus, corpus_paragraphs="", file_open_mode="w", sep = "\n\n")
    print("article.save_paragraphs() fini")


def test_datasource(url, topic, num_articles):
    datasource = DataSource(url, topic, num_articles)
    print(datasource)


def test_bibliography(url, topic, num_articles):
    bibliography = Bibliography(url, topic, num_articles)
    print(bibliography)
    bibliography.save_paragraphs(savemode="overwrite")
    # print("bibliography.save_paragraphs() fini")


def test_bibliographylist(urls, num_articles, topic="filenames"):
    bibliographyList = BibliographyList(urls, num_articles, topic)
    print(bibliographyList)
    # bibliographyList.save_articles_urls()
    bibliographyList.save_paragraphs()


def test_blog(url, num_articles):
    blog = Blog(url, num_articles)
    print(blog)
    # blog.save_articles_urls()
    # blog.save_paragraphs(savemode="overwrite")


def test_bloglist(urls, num_articles):
    blogList = BlogList(urls, num_articles)
    print(blogList)
    # blogList.save_articles_urls()
    blogList.save_paragraphs()


def main():
    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    # url = "https://parlafoi.fr/lire/series/le-pedobapteme/"
    url = "https://parlafoi.fr/lire/series/la-scolastique-protestante/"
    url2 = "https://parlafoi.fr/lire/series/la-scolastique-protestante/"
    url3 = "https://parlafoi.fr/lire/series/commentaire-de-la-summa/"
    topic = "moyen_age"
    num_articles = 40
    # num_articles = "all"

    # test_article(url, topic)
    # test_datasource(url, topic, num_articles)
    # test_bibliography(urls_biblio, topic, num_articles)
    # test_blog(url_blog, num_articles)

    # urls_biblio = [url2, url3]
    # urls_biblio = "bibliography_middle_age_fr.txt"
    # topic = "filenames"
    # test_bibliographylist(urls_biblio, num_articles, topic)
    # test_bibliographylist(urls_biblio, num_articles)

    # urls_blog = ["http://exapologist.blogspot.com", "http://alexanderpruss.blogspot.com"]
    urls_blog = "blogs_philosophy_eng.txt"
    # urls_blog = ["http://exapologist.blogspot.com"]
    num_articles = 4
    test_bloglist(urls_blog, num_articles)

    # bibliography.

main()