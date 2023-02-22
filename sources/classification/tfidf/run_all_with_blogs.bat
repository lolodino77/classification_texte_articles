@REM Script .bat pour executer toutes les etapes du projet sur Windows avec des blogs en entree

ECHO OFF
@REM 
set list_of_blogs=%1
set list_corpus_topic=%2
set num_articles=%3
set corpus_extension=%4
set merged_corpus_extension=%5
set language=%6
echo %list_of_blogs%
echo %list_corpus_topic%
echo %num_articles%
echo %corpus_extension%
echo %merged_corpus_extension%
echo %language%

@REM Partie pretraitement des donnees
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles\sources\create_dataset
python 0_create_one_topic_corpus.py %list_of_blogs% %num_articles% %corpus_extension%
python 1_create_multiple_topic_corpus.py %list_corpus_topic% %language% %merged_corpus_extension%

@REM Partie modele 
@REM cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles\sources\classification\tfidf
python 2_model_selection.py corpus_sceptic_theist.%merged_corpus_extension%
python 3_train_test_best_model.py corpus_sceptic_theist.%merged_corpus_extension%

@REM cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles\sources
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles\sources\classification\tfidf
rem Fin

@REM Exemple de valeurs pour les parametres
@REM set blogs_list_filename=blogs_philosophy.txt
@REM list_corpus_topic=list_corpus_philosophy.txt
@REM set num_articles=15
@REM set corpus_extension=csv
@REM set corpus_extension=parquet
@REM set merged_corpus_extension=csv
@REM set merged_corpus_extension=parquet
@REM set language=english

@REM Exemple de commande de terminal windows
@REM run_all_with_blogs.bat blogs_philosophy.txt list_corpus_philosophy.txt 15 csv csv english
@REM run_all_with_blogs.bat blogs_list_filename num_articles corpus_extension merged_corpus_extension language

@REM python 0_create_one_topic_corpus.py bibliography_philosophie.txt 5 csv
@REM python 0_create_one_topic_corpus.py little_blogs.txt 60 csv
@REM python 1_create_multiple_topic_corpus.py list_corpus_feser_pruss.txt english csv
@REM python 2_model_selection.py ./data/input/merged_corpus/ csv
