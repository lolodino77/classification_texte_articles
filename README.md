# Classification binaire de textes (articles de blogs wordpress et blogspot)

## Problème : classer des articles de blog dans des catégories (soit des auteurs, soit des thèmes/topics)
Il s'agit d'un projet personnel de NLP qui vise à implémenter un classifier binaire (à deux classes) de textes d'articles de blogs. Son but sera de reconnaître à partir d'un article de blog soit son auteur (nom prénom, pseudo ou nom du blog d'origine), soit de reconnaître son topic (par exemple science ou littérature). Dans le premier cas les classes sont des auteurs, dans le second des topics.  
## Méthodes de classification en NLP utilisées
Nous testerons au fur et à mesure les principales méthodes de classification en NLP (pour chacune nous indiquons les étapes achevées) :
	* Méthodes basées sur la fréquence des mots (tfidf, observed-expected, PMI, etc.) / modularisation du code & dockeurisation finie
	* Méthodes basées sur la représentation des mots dans un espace vectoriel (word2vec : cbow et skip-gram) / notebook fini
	* Méthodes basées sur des réseaux de neurones (transformers, etc.) / à commencer

## Les différentes étapes du projet de l'extraction des ressources à la mise en production
* Web-scraping avec beautifulSoup des articles de blogs wordpress et blogspot
* Création du corpus de textes et annotation des données avec pandas
* Prétraitement du corpus : majuscules, ponctuation, stopwords, tokenisation avec nltk
* Réduction de dimension (svd, tsne, umap avec sklearn) et visualisation (matplotlib et
* seaborn)
* Feature engineering (tfidf, observed-expected, PMI et word2vec – sklearn et nltk)
* Sélection de modèles : cross-validation, learning curves
* Version de départ en notebooks .ipynb
* Version de mise en production en scripts python .py
* Modularisation (programmation orientée objet) des scripts pour une meilleure maintenance et reproductibilité (pratiques de "software engineering")
* Création de scripts windows .bat pour tester facilement l'ensemble projet
* Conteneurisation du code avec Docker pour faciliter le déploiement et l’utilisation de tiers

## Dossiers
### **data** : contient toutes les données du projet (entrées et sorties)
	* input : contient les données d'entrées (initiales et intermédiaires, c'est-à-dire des transformations des données initiales)
	* output : contient les sorties des codes de machine learning (des mesures de performance des modèles comme des matrices de confusion, des performances de k-fold cross-validation, des learning curves, etc.)
### **sources** : les codes du projet
	* **create_dataset** : les codes pour créer les datasets (des corpus à plusieurs topics, appelés dans ce projet "merged_corpus") donnés en entrée aux modèles de classification
	* **dimension_reduction_visualisation** : les codes réduire les dimensions des corpus et les visualiser en 2D/3D
	* **classification** : les codes pour implémenter les modèles de classification
		* **count_matrix** : le code qui implémente la méthode du nombre d'occurrences de chaque terme
			* 1_count_matrix.ipynb : le notebook pour visualiser et analyser la matrice tfidf 
		* **tfidf** : le code qui implémente la méthode tfidf
			* 1_tfidf.ipynb : le notebook pour visualiser et analyser la matrice tfidf 
		* **word2vec** le code qui implémente les méthodes de type word2vec (cbow et skip-gram)  
		* Dans chacun de ces sous-dossiers :
			* 2_model_selection.py : script qui implémente la sélection des modèles par k-fold cross-validation et learning curves
			* 3_train_test_best_model.py : script qui entraîne et évalue le meilleur modèle sélectionné suite à la sélection de modèles


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
