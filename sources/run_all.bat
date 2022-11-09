ECHO OFF
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources\create_dataset
python 0_create_individual_topic_corpus.py ./data/input/bibliographies/ txt
python 1_create_final_corpus.py dataset_baptism_fr.csv dataset_philosophy_fr.csv

cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources\classification\tfidf
python 2_model_selection.py ./data/input/ parquet
cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources
rem Fin