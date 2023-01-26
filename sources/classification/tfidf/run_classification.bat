@REM Script .bat pour executer l'etape de classification du projet sur Windows
    @REM 2_model_selection.py : selection de modele par k-fold cross-validation     
    @REM 3_train_test_best_model.py : entraine et test le meilleur modele

ECHO OFF
python 2_model_selection.py ./data/input/ parquet
python 3_train_test_best_model.py corpus_feser_pruss.csv
python 3_train_test_best_model.py corpus_sceptic_theist.csv