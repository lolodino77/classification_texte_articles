@REM Script .bat pour executer l'etape de creation de corpus a plusieurs topics sur Windows

ECHO OFF
python 1_create_multiple_topic_corpus.py corpus_list_philosophy.txt english csv
python 1_create_multiple_topic_corpus.py corpus_list_feser_pruss.txt english csv

rem Fin