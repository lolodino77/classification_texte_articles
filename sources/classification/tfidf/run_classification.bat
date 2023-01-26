@REM Script .bat pour executer l'etape de classification du projet sur Windows

ECHO OFF
python 2_model_selection.py ./data/input/ parquet
python 3_train_test_best_model.py corpus_feser_pruss.csv
python 3_train_test_best_model.py corpus_sceptic_theist.csv