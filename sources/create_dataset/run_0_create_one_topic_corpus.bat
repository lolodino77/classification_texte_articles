@REM Script .bat pour executer l'etape de creation de corpus a un seul topic sur Windows
@REM Pattern :
@REM python 0_create_one_topic_corpus.py <source_filename> <num_articles> <output_extension>

@REM Exemple :
@REM python 0_create_one_topic_corpus.py bibliography_philosophie.txt 5 csv
    @REM <source_filename> = bibliography_philosophie.txt
    @REM <num_articles> = 5
    @REM <output_extension> = csv 

ECHO OFF
python 0_create_one_topic_corpus.py bibliography_philosophie.txt 5 csv
python 0_create_one_topic_corpus.py blogs_philosophy.txt 15 csv
python 0_create_one_topic_corpus.py little_blogs.txt 60 csv

python 0_create_one_topic_corpus.py blogs.txt 10 csv
rem Fin