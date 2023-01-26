ECHO OFF
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\classification\tfidf
python 2_model_selection.py ./data/input/merged_corpus/ csv
python 2_model_selection.py all parquet
python 2_model_selection.py corpus_edwardfeser_exapologist.parquet
python 2_model_selection.py all csv
python 2_model_selection.py corpus_edwardfeser_exapologist.csv

cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources
rem Fin