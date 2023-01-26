@REM Script .bat pour executer l'etape de selection de modeles (cross-validation) sur Windows

ECHO OFF
python 2_model_selection.py ./data/input/merged_corpus/ csv
python 2_model_selection.py all parquet
python 2_model_selection.py corpus_edwardfeser_exapologist.parquet
python 2_model_selection.py all csv
python 2_model_selection.py corpus_edwardfeser_exapologist.csv

rem Fin