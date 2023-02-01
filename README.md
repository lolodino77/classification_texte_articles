# Classification binaire de textes (articles de blogs wordpress et blogspot)

## Problème : classer des articles de blog dans des catégories (soit des auteurs, soit des thèmes/topics)
Il s'agit d'un projet personnel qui vise à implémenter un classifier binaire (à deux classes) de textes d'articles de blogs. Son but sera à partir d'un article de blog soit de reconnaître son auteur (nom prénom, pseudo ou nom du blog d'origine), soit de reconnaître son topic (par exemple science ou littérature). Dans le premier cas les classes sont des auteurs, dans le second des topics.  
## Dossiers
* **data** : contient toutes les données du projet (entrées et sorties)
	* input : contient les données d'entrées (initiales et intermédiaires, c'est-à-dire des transformations des données initiales)
	* output : contient les sorties des codes de machine learning (résultats comme des matrices de confusion, des performances de k-fold cross-validation, des learning curves, etc.)
* **mpi_et_openmp** : algorithme avec MPI et OpenMP
* **certificats** : certificats qui prouvent l'intégrité de la solution trouvée (qu'on n'a pas "triché" mais bien calculer la solution avec un algorithme)

## Fichiers
* **Makefile** : makefile qui pour compiler le programme
* **checkpoint** : valeurs sauvegardées lors du dernier calcul interrompu (peut être importer par le programme pour éviter de recommencer le calcul)
* **hostile, hostfile2 et hostfile_single** : liste des machines/processus auquel on peut faire appel lors du calcul
* **res** : solution du système obtenue après calcul 



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
Cree les parametres du modele d'apprentissage supervise
