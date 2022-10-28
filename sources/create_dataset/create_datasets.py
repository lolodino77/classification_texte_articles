from lib_scraping import *

# Creation du dataframe philosophy
# bibliography_urls = ["https://parlafoi.fr/lire/series/commentaire-de-la-summa/",
# "https://parlafoi.fr/lire/series/notions-de-base-en-philosophie/",
# "https://parlafoi.fr/lire/series/la-scolastique-protestante/",
# "https://parlafoi.fr/lire/series/le-presuppositionnalisme/"]
# filename_urls_articles = "urls_philosophy_articles.txt"
# filename_corpus = "corpus_philosophy.txt"

# write_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode="a")

# filename_corpus_input = "corpus_philosophy.txt"
# filename_corpus_output = "dataset_philosophy.csv"

# write_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode="a")


# Creation du dataframe baptism
# bibliography_urls = ["https://parlafoi.fr/lire/series/le-pedobapteme/"]
# filename_urls_articles = "urls_baptism_articles.txt"
# filename_corpus = "corpus_baptism.txt"

# write_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode="a")

# filename_corpus_input = "corpus_baptism.txt"
# filename_corpus_output = "dataset_baptism.csv"

# write_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode="a")


# Creation du dataframe epistemology
bibliography_urls = ["https://parlafoi.fr/lire/series/le-presuppositionnalisme/"]
filename_urls_articles = "urls_epistemology_articles.txt"
filename_corpus = "corpus_epistemology.txt"

write_corpus_from_urls(bibliography_urls, filename_urls_articles, filename_corpus, file_open_mode="a")

filename_corpus_input = "corpus_epistemology.txt"
filename_corpus_output = "dataset_epistemology.csv"

write_corpus_dataset_from_paragraphs(filename_corpus_input, filename_corpus_output, file_open_mode="a")