@REM Script .bat pour executer toutes les etapes du projet sur Windows

ECHO OFF
@REM Partie pretraitement des donnees
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\create_dataset
@REM python 0_create_one_topic_corpus.py bibliography_philosophie.txt 5 csv
python 0_create_one_topic_corpus.py blogs_philosophy.txt 15 csv
@REM python 0_create_one_topic_corpus.py little_blogs.txt 60 csv
python 1_create_multiple_topic_corpus.py list_corpus_philosophy.txt english csv
@REM python 1_create_multiple_topic_corpus.py list_corpus_feser_pruss.txt english csv

@REM Partie modele 
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\classification\tfidf
@REM python 2_model_selection.py ./data/input/merged_corpus/ csv
python 2_model_selection.py corpus_sceptic_theist.csv csv
python 3_train_test_best_model.py corpus_sceptic_theist.csv

cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources
rem Fin