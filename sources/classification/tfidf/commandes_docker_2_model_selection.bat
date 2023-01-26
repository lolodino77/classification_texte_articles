cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet

@REM Exemples de commandes pour lancer le .bat
@REM Syntaxe generale :
@REM commandes_docker_2_model_selection <corpus_name> <corpus_extension>
@REM <corpus_name> : nom du corpus (son sujet)
@REM <corpus_extension> : extension du corpus (csv ou parquet)

@REM Exemples particuliers :
@REM pour lancer le script sur le corpus chien_chat.csv, ecrire :
@REM commandes_docker_2_model_selection chien_chat csv
@REM commandes_docker_2_model_selection sceptic_theist csv
@REM commandes_docker_2_model_selection corpus_alexanderpruss_edwardfeser parquet

set corpus_name=%1
set corpus_extension=%2
@REM set corpus_name=feser_pruss
@REM set corpus_name=sceptic_theist
@REM set corpus_extension=csv

docker rm -f conteneur_tfidf_classif
docker image rm image_tfidf_classif
docker build --build-arg corpus_name --build-arg corpus_extension -t image_tfidf_classif -f sources/classification/tfidf/Dockerfile_2_model_selection .
docker run --name=conteneur_tfidf_classif -d -v %cd%\data\output\%corpus_name%:/classification_texte_articles_version_objet/data/output/%corpus_name%/ image_tfidf_classif corpus_%corpus_name%.%corpus_extension%
docker logs conteneur_tfidf_classif

@REM docker start -i conteneur_tfidf_classif

cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet\sources\classification\tfidf