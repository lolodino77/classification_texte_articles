ECHO OFF
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\create_dataset
python 1_create_multiple_topic_corpus.py corpus_list_philosophy.txt english csv
python 1_create_multiple_topic_corpus.py corpus_list_feser_pruss.txt english csv

cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources
rem Fin