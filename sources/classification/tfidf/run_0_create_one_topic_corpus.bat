ECHO OFF
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\create_dataset
python 0_create_one_topic_corpus.py bibliography_philosophie.txt 5 csv
python 0_create_one_topic_corpus.py blogs_philosophy.txt 15 csv
python 0_create_one_topic_corpus.py little_blogs.txt 60 csv
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources
rem Fin