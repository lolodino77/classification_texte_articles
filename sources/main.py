from article import *
from bibliography import *
import os


def test_article(url, topic):
    article = Article(url, topic)
    print(article)
    article.save_paragraphs(file_open_mode="w", sep = "\n\n")
    print("article.save_paragraphs() fini")


def test_bibliography(url, topic, num_articles):
    bibliography = Bibliography(url, topic, num_articles)
    print(bibliography)
    bibliography.save_paragraphs(savemode="overwrite")
    print("bibliography.save_paragraphs() fini")


def main():
    set_current_directory_to_root(root = "classification_texte_articles_version_objet")
    print("os.getcwd()")
    print(os.getcwd())
    url = "https://parlafoi.fr/lire/series/le-pedobapteme/"
    topic = "moyen_age"
    # num_articles = 6
    num_articles = "all"

    test_bibliography(url, topic, num_articles)

    # bibliography.

main()