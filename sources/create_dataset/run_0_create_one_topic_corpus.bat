@REM Script .bat pour executer l'etape de creation de corpus a un seul topic sur Windows

ECHO OFF
python 0_create_one_topic_corpus.py bibliography_philosophie.txt 5 csv
python 0_create_one_topic_corpus.py blogs_philosophy.txt 15 csv
python 0_create_one_topic_corpus.py little_blogs.txt 60 csv
rem Fin