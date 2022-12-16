ECHO OFF
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\classification\tfidf
@REM python 2_model_selection.py ./data/input/ parquet
@REM python 3_train_test_best_model.py corpus_feser_pruss.csv
python 3_train_test_best_model.py corpus_sceptic_theist.csv
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources