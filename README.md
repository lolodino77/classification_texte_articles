
# Classification binaire de textes (articles de blogs wordpress et blogspot)

## Résumé
Il s'agit d'un projet personnel de NLP qui vise à implémenter un classifier binaire (à deux classes) de textes d'articles de blogs. Son but sera de reconnaître à partir d'un article de blog soit son auteur (nom prénom, pseudo ou nom du blog d'origine), soit de reconnaître son topic (par exemple science ou littérature). Dans le premier cas les classes sont des auteurs, dans le second des topics.  
## Méthodes de classification en NLP utilisées
Nous testerons au fur et à mesure les principales méthodes de classification en NLP (pour chacune nous indiquons les étapes achevées) :
	* Méthodes basées sur la fréquence des mots (tfidf, observed-expected, PMI, etc.) / modularisation du code & dockeurisation finie
	* Méthodes basées sur la représentation des mots dans un espace vectoriel (word2vec : cbow et skip-gram) / notebook fini
	* Méthodes basées sur des réseaux de neurones (transformers, etc.) / à commencer

## Etapes du projet du web-scraping à la mise en prod
* **Web-scraping** avec beautifulSoup des articles de blogs wordpress et blogspot
* **Création du corpus de textes** et **annotation** des données avec pandas
* **Prétraitement du corpus** : majuscules, ponctuation, stopwords, tokenisation avec nltk
* **Réduction de dimension** (svd, tsne, umap avec sklearn) et **visualisation** (matplotlib et seaborn)
* **Feature engineering** (tfidf, observed-expected, PMI et word2vec – sklearn et nltk)
* **Sélection de modèles** par cross-validation et learning curves
* **Version de départ** en notebooks .ipynb
* **Version de mise en production** en scripts python .py
* **Modularisation** et création de classes (**programmation orientée objet)** des scripts pour une meilleure maintenance et reproductibilité (pratiques de "software engineering")
* **Création de scripts windows .bat** pour tester facilement l'ensemble projet
* **Conteneurisation du code avec Docker** pour faciliter le déploiement et l’utilisation de tiers

## Dossiers et codes
### data : les données du projet
* **input** : les données d'entrées (initiales et intermédiaires, c'est-à-dire des transformations des données initiales)
* **output** : les sorties des codes de machine learning (des mesures de performance des modèles comme des matrices de confusion, des performances de k-fold cross-validation, des learning curves, etc.)
### sources : les codes du projet
#### create_dataset : création des datasets
* *datasource.py*, *bibliography.py*, *blog.py*, *wordpress.py*, *blogspot.py*, *datasourcelist.py*, *bloglist.py*, *bibliographylist.py*, *article.py* : définition des classes (un script par classe), POO
* *0_create_one_topic_corpus.py* : crée un corpus à un topic
* *1_create_multiple_topic_corpus* : crée un corpus à plusieurs topics (en fusionnant deux corpus à un topic) appelé "merged_corpus" donné en entrée aux modèles de classification
#### dimension_reduction_visualisation : réduit les dimensions des corpus et les visualise en 2D/3D
* *dimension_reduction.ipynb*
#### classification : les codes pour implémenter les modèles de classification
* **count_matrix** : la méthode basée sur le nombre d'occurrences de chaque terme
	* *1_count_matrix.ipynb* : le notebook pour visualiser et analyser la matrice tfidf 
* **tfidf** : méthode tfidf
	* *1_tfidf.ipynb* : le notebook pour visualiser et analyser la matrice tfidf
	* *requirements.txt* : dépendances python à installer
	* *Dockerfile_2_model_selection* : Dockerfile pour lancer le script 2_model_selection.py 
	* *Dockerfile_3_train_test* : Dockerfile pour lancer le script 3_train_test_best_model.py
	* *run_docker_2_model_selection.bat* : Executable windows qui crée l'image et le conteneur qui execute 2_model_selection.py a partir de Dockerfile_2_model_selection
	* *run_docker_3_train_test.bat* : Executable windows qui crée l'image et le conteneur qui execute 3_train_test_best_model.py a partir de Dockerfile_3_train_test
	* *run_all_with_blogs.bat* : Executable windows qui lance tous les scripts de la création des corpus à la classification
	* *run_classification.bat* : Executable windows qui lance les scripts de classification (*2_model_selection.py* et *3_train_test_best_model.py*)
* **word2vec** : méthodes de type word2vec (cbow et skip-gram)  
	* *0_word2vec_cbow_feature_extraction.ipynb* :  
	* *0_word2vec_skip_gram_feature_extraction.ipynb* :  
	* *1_word2vec_cbow_classification.ipynb* : classification cbow
	* *1_word2vec_skip_gram_classification.ipynb* : classification skip-gram  
	* *1_word2vec_glove.ipynb* : classification glove
* Dans chacun de ces sous-dossiers :
	* *2_model_selection.py* : script qui implémente la sélection des modèles par k-fold cross-validation et learning curves
	* *3_train_test_best_model.py* : script qui entraîne et évalue le meilleur modèle sélectionné suite à la sélection de modèles

## Environnement Technique
* **Editeurs de texte/IDE** : Sublime Text, VS Code
* **Logique de programmation** : Programmation orientée objet (POO)
* **Langage** : Python
* **Librairies** : BeautifulSoup, NLTK, Sklearn, Matplotlib, Seaborn, Numpy, Pandas

<!-- 
-------------------------------codes dans l'ordre d'execution-----------------------------------
lib_scraping.py :
N'est pas execute dans le terminal mais contient la librarie avec toutes les fonctions utilisees par les autres fichiers .py

get_corpus_philosophy.py :
Ecrit dans le fichier texte corpus_philosophy.txt les parties de chaque texte du corpus de philosophie apres l'avoir decoupe

get_corpus_baptism.py :
Ecrit dans un fichier texte corpus_baptism.txt les parties de chaque texte du corpus sur le bapteme apres l'avoir decoupe

create_dataset_philosophy.py :
Cree le dataframe pour l'algorithme d'apprentissage automatique stocke dans le fichier dataset_philosophy.csv

create_dataset_baptism.py :
Cree le dataframe pour l'algorithme d'apprentissage automatique stocke dans le fichier dataset_baptism.csv

0_preprocess_corpus.py :
Pretraite les messages du corpus pour les renvoyer dans un format exploitable par les algo d'IA

1_feature_engineering.py :
Cree les parametres du modele d'apprentissage supervise -->
