from article import *
from bibliography import *
import os


def test_article(url, topic):
    article = Article(url, topic)
    print(article)
    path_corpus = "./data/input/corpus_txt/corpus_{}.txt".format(topic)  #self.filename
    article.save_paragraphs(path_corpus, file_open_mode="w", sep = "\n\n")
    print("article.save_paragraphs() fini")


def test_datasource(url, topic, num_articles):
    datasource = DataSource(url, topic, num_articles)
    print(datasource)


def test_bibliography(url, topic, num_articles):
    bibliography = Bibliography(url, topic, num_articles)
    print(bibliography)
    bibliography.save_paragraphs(savemode="overwrite")
    # print("bibliography.save_paragraphs() fini")


# def test_blog(url, topic, num_articles):
#     blog = Blog(url, topic, num_articles)
#     print(blog)
    # blog.save_paragraphs(path_corpus, savemode="overwrite")
    # print("blog.save_paragraphs() fini")


def main():
    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    # url = "https://parlafoi.fr/lire/series/le-pedobapteme/"
    # url = "https://parlafoi.fr/lire/series/la-scolastique-protestante/"
    url = "https://parlafoi.fr/lire/series/commentaire-de-la-summa/"
    topic = "moyen_age"
    # num_articles = 6
    num_articles = "all"

    # test_article(url, topic)
    # test_datasource(url, topic, num_articles)
    test_bibliography(url, topic, num_articles)

    # bibliography.

main()