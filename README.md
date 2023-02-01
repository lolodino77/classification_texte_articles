# Projet de calcul haute performance en première année d'école d'ingénieur

## Problème : résolution de système linéaire avec matrices creuses
Il s'agit d'un projet qui vise à appliquer les outils de calcul parallèle (OpenMPI et OpenMP) vu en cours de calcul haute performance (HPC) à un projet en binôme de résolution de système linéaire (avec des matrices creuses) dans le cadre d'un projet scolaire en binôme.
Explication du code dans le PDF des consignes du projet.

## Dossiers
* **mpi** : algorithme avec MPI
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
