@REM Script .bat pour executer l'etape de creation de corpus a plusieurs topics sur Windows
@REM Pattern :
@REM python 0_create_one_topic_corpus.py <filename_corpus_list> <language> <output_extension>
    @REM <filename_corpus_list> : nom du fichier liste des corpus a fusionner avec leur topic
    @REM <language> : langue utilisee (english ou french)
    @REM <output_extension> : l'extension du nouveau corpus cree par la fusion des corpus 

@REM Exemple :
@REM python 1_create_multiple_topic_corpus.py corpus_list_philosophy.txt english csv
    @REM <filename_corpus_list> = corpus_list_philosophy.txt
        @REM corpus_alexanderpruss.txt theist
            @REM corpus_edwardfeser.txt theist
            @REM corpus_majestyofreason.txt sceptic
            @REM corpus_exapologist.txt sceptic
    @REM <language> = english
    @REM <output_extension> = csv

ECHO OFF
python 1_create_multiple_topic_corpus.py list_corpus_philosophy.txt english csv
python 1_create_multiple_topic_corpus.py list_corpus_feser_pruss.txt english csv

rem Fin