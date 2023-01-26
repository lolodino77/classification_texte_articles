@REM Script .bat pour executer toutes les etapes du projet sur Windows

ECHO OFF
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\create_dataset
@REM python 0_create_one_topic_corpus.py bibliography_philosophie.txt 5 csv
@REM python 0_create_one_topic_corpus.py blogs_philosophy.txt 15 csv
@REM python 0_create_one_topic_corpus.py little_blogs.txt 60 csv
@REM python 1_create_multiple_topic_corpus.py corpus_list_philosophy.txt english csv
python 1_create_multiple_topic_corpus.py corpus_list_feser_pruss.txt english csv

cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\classification\tfidf
python 2_model_selection.py ./data/input/merged_corpus/ csv
@REM python 3_train_test_best_model.py corpus_sceptic_theist.csv
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources
rem Fin