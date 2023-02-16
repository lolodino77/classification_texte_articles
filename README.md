# Classification binaire de textes

## Résumé
Il s'agit d'un projet personnel en deux parties de NLP en python qui vise à implémenter un classifier binaire (à deux classes) de textes. On fera aussi du web scraping, du nettoyage de données (ici du texte) et on déploiera la solution avec Docker.

### Partie 1 : Classification d'articles de blogs wordpress et blogspot
en particulier d'articles de blogs. Mais il reste adaptable à tout corpus binaire. Son but sera de reconnaître à partir d'un paragraphe d'un article de blog :
<ol>	
<li>Soit son auteur (nom prénom, pseudo ou nom du blog d'origine). Dans ce cas les classes sont des auteurs. Par exemple : Edward Feser ou Joe Schmid, Alexander Pruss ou Felipe Leon.</li>
<li>Soit son topic (par exemple science ou littérature). Dans ce cas les classes sont des topics. Par exemple : science ou littérature, actualités ou cuisine.</li>  
<li>Soit sa prise de position (par exemple vegan ou non vegan, pro apple ou pro android). Dans ce cas les classes sont des prises de position.
</ol>

Autrement dit, dans cette première partie, un document (ou message à classer dans une catégorie) sera un paragraphe d'un article de blog.

Le but était aussi d'avoir un outil qui crée rapidement, automatiquement et facilement des datasets à partir d'articles de blogs wordpress ou blogspot pour de la classification binaire NLP. Il suffit pour cela de donner en entrées des url de pages d'accueil de blogs (par exemple www.blog.wordpress.com) ou des url de pages "bibliographies" qui listent des urls (des articles de blogs wordpress ou blogspot).

### Partie 2 : Application sur un business case (classification de reviews Amazon)
Dans cette seconde partie, nous appliquons notre solution à un cas pratique de business. Ici, classer des avis sur des produits Amazon soit dans la classe positive (en cas de satisfaction du client), soit la classe négative (en cas de déception du client). En effet, l'application est codée de façon à aussi pouvoir prendre en entrée des corpus à deux classes déjà prétraités (avec des messages bruts ou déjà nettoyés).
Les différentes étapes :
<ol>	
<li>Le choix du corpus : le corpus amazon_reviews_us_Electronics_v1_00 (des avis et des notes de 1 à 5 étoiles sur des produits d'électronique) téléchargeable à ce lien : https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt</li>
<li>Le prétraitement du corpus d'origine : enlever les colonnes inutiles, sélection des lignes à garder (lignes avec une note égale à 1 ou 5), création des annotations (annotation positive pour les notes à 5, négative pour celles à 1).</li>  
<li>La partie IA : sélection du meilleur modèle par cross-validation & learning curves, entraînement et évaluation de ce modèle, interprétation des résultats.
</ol>

## Enjeux du projet : pourquoi des blogs ?
* A notre époque, l'effervescence se concentre désormais principalement sur les réseaux sociaux (instagram, TikTok, Facebook, Discord) au détriment des supports majeurs de mon adolescence qu'étaient les blogs (Skyblog, OverBlog, Eklablog, E-monsite) et les forums (ex : Xooit, Forumactif). Pourquoi alors réaliser un projet de NLP porté sur les blogs ?
* Si beaucoup de jeunes des génération Y et Z ne partagent leur contenu que sur les réseaux sociaux, ce n'est pas le cas des débuts génération Z et de tous les Y, et encore moins pour les moins jeunes.
* Les blogs restent notamment une plateforme de vulgarisation privilégiée pour les personnes passionées par un sujet. Par exemple : l'histoire, l'actualité, la politique, la philosophie ou encore la data science ! C'est parce qu'ils permettent à la fois de partager un contenu à la fois plus poussé que les publications sur des réseaux sociaux et youtube, et gratuit contrairement aux livres et aux articles de recherche. 
* Dans mon cas, mon objectif personnel était de travailler sur des datasets de philosophie. Les blogs s'y prêtaient bien : on y trouve beaucoup de chercheurs et d'étudiants en philosophie dans le milieu anglo-saxon. En cela, ils demeurent donc une source gratuite idéale pour créer des datasets de philosophie. Par exemple voir les blogs d'Edward Feser, d'Alexander Pruss, de Felipe Leon et de Joe Schmid (quatre de mes philosophes préférés que je suis régulièrement) :
	* http://edwardfeser.blogspot.com/
	* http://alexanderpruss.blogspot.com/
	* http://exapologist.blogspot.com/ (Felipe Leon)
	* https://majestyofreason.wordpress.com/ (Joe Schmid)

## Environnement Technique
* **Editeurs de texte/IDE** : Sublime Text, VS Code
* **Logique de programmation** : Programmation orientée objet (POO)
* **Langage** : Python
* **Librairies** : BeautifulSoup, NLTK, Sklearn, Matplotlib, Seaborn, Numpy, Pandas
* **Mise en production** : Docker

## Méthodes de classification en NLP utilisées
Nous testerons au fur et à mesure les principales méthodes de classification en NLP (pour chacune nous indiquons les étapes achevées) :
<ol>
<li>Méthodes basées sur la fréquence des mots (tfidf, observed-expected, PMI, etc.) / modularisation du code & dockeurisation finie</li>
<li>Méthodes basées sur la représentation des mots dans un espace vectoriel (word2vec : cbow et skip-gram) / notebook fini</li>
<li>Méthodes basées sur des réseaux de neurones (transformers, etc.) / à commencer</li>
</ol>

## Etapes du projet du web-scraping à la mise en prod
* **Web-scraping** avec beautifulSoup des articles de blogs wordpress et blogspot
* **Création du corpus de textes** et **annotation** des données avec pandas
* **Prétraitement du corpus** : majuscules, ponctuation, stopwords, tokenisation avec nltk
* **Réduction de dimension** (svd, tsne, umap avec sklearn) et **visualisation** (matplotlib et seaborn)
* **Feature engineering** (tfidf, observed-expected, PMI et word2vec – sklearn et nltk)
* **Sélection de modèles** par cross-validation et learning curves
* **Version de tests** en notebooks .ipynb
* **Version de mise en production** en scripts python .py
* **Modularisation** et création de classes (**programmation orientée objet)** des scripts pour une meilleure maintenance et reproductibilité (pratiques de "software engineering")
* **Création de scripts windows .bat** pour tester facilement l'ensemble projet
* **Conteneurisation du code avec Docker** pour faciliter le déploiement et l’utilisation de tiers (avec utilisation de la notion de volume pour persister les fichiers de sortie des conteneurs)

## Dossiers et codes
Les fichiers .py sont les versions de mise en production. Ils sont donc a minima structurés en module ou en classe (programmation orientée objet) pour une meilleure reproductivité et facilité d'utilisation.

Les fichiers .ipynb sont :
<ol>
<li>Soit des codes qui font de l'analyse exploratoire/statistique des données (étape préliminaire)</li>
<li>Soit des versions de tests avant mise en prod (scripts .py).</li>
</ol>

De façon générale, pour exécuter le projet dans des conditions réelles, on utilisera les scripts de terminaux (soit avec des commandes python du type "python file.py ...", soit avec Docker). Pour voir plus rapidement des exemples et résultats du projet, on pourra consulter directement les notebooks.

### data : les données du projet
* **input** : les données d'entrées (initiales et intermédiaires, c'est-à-dire des transformations des données initiales)
	* **blogs** : dossier avec les fichiers .txt listes de blogs
	* **bibliographies** : dossier avec les fichiers .txt bibliographies
	* **articles_lists** : dossier avec les fichiers .txt liste d'articles (sortie du script *0_create_one_topic_corpus.py*)
	* **corpus_txt** : dossier avec les fichiers .txt de corpus, une suite de messages/paragraphes (sortie du script *0_create_one_topic_corpus.py*)
	* **corpus_csv** : dossier avec les corpus à un topic au format .csv (avec comme colonnes : id + message + topic)
	* **corpus_parquet** : dossier avec les corpus à un topic au format .parquet (avec comme colonnes : id + message + topic)
	* **merged_corpus** : dossier avec les corpus à deux topics au format .csv ou .parquet (colonnes : id + message + topic)

* **output** : les sorties des codes de machine learning (des mesures de performance des modèles comme des matrices de confusion, des performances de k-fold cross-validation, des learning curves, etc.)
	* **<dataset_name>** : crée un sous-dossier pour chaque dataset sur lequel on effectue la classification (le nom du dossier <dataset_name> est le même que le nom du fichier qui stocke le dataset, <dataset_name> dans <dataset_name>.csv ou <dataset_name>.parquet)
		* **select_model** : dossier avec les sorties de la sélection de modèles (résultats de la k-fold cross-validation et learning curves)
		* **best_model** : dossier avec les sorties de l'évaluation des performances du meilleur modèle retenu après sélection des modèles (matrice de confusion, classification_report de sklearn, learning curves et ) 

### sources : les codes du projet
* *lib_general.py* : module qui contient des fonctions "générales" utiles durant différentes étapes du projet
	#### create_dataset : création des datasets
	* *datasource.py*, *bibliography.py*, *blog.py*, *wordpress.py*, *blogspot.py*, *datasourcelist.py*, *bloglist.py*, *bibliographylist.py*, *article.py* : définition des classes (un script par classe), POO
	* *0_create_one_topic_corpus.py* : crée un corpus à un topic
	* *1_create_multiple_topic_corpus* : crée un corpus à plusieurs topics (en fusionnant deux corpus à un topic) appelé "merged_corpus" donné en entrée aux modèles de classification
	* 	*run_0_create_one_topic_corpus.bat* : exécutable windows qui lance le script *0_create_one_topic_corpus.py*
	* *run_1_create_multiple_topic_corpus.bat* : exécutable windows qui lance tous le script *1_create_multiple_topic_corpus*
	#### dimension_reduction_visualisation : réduit les dimensions des corpus et les visualise en 2D/3D
	* *dimension_reduction.ipynb*
	#### classification : modèles de classification
	* *lib_classification.py* : module qui contient des fonctions pour classification (ex : fonction pour la sélection des modèles, fonction pour la matrice de confusion, fonction pour les learning curves, etc.)
	* *test_transformations_matrice_count.ipynb* : notebook d'exploration qui teste et compare les differentes méthodes transformations de matrices de fréquences (tfidf, observed-expected, PMI, etc.), le but étant d'en trouver une qui donne une matrice "riches en valeurs", avec une distribution plus riche en valeurs possibles et au mieux une distribution gaussienne   
	* **count_matrix** : dossier pour la méthode basée sur le nombre d'occurrences de chaque terme
		* *1_count_matrix.ipynb* : le notebook pour visualiser et analyser la matrice tfidf 
	* **tfidf** : méthode tfidf
		* *1_tfidf.ipynb* : notebook pour visualiser et analyser la matrice tfidf
		* *requirements.txt* : dépendances python à installer
		* *Dockerfile_2_model_selection* : Dockerfile pour lancer le script 2_model_selection.py 
		* *Dockerfile_3_train_test* : Dockerfile pour lancer le script 3_train_test_best_model.py
		* *run_docker_2_model_selection.bat* : exécutable windows qui crée l'image et le conteneur qui execute 2_model_selection.py a partir de Dockerfile_2_model_selection
		* *run_docker_3_train_test.bat* : exécutable windows qui crée l'image et le conteneur qui execute 3_train_test_best_model.py a partir de Dockerfile_3_train_test
		* *run_all_with_blogs.bat* : exécutable windows qui lance tous les scripts de la création des corpus à la classification
		* *run_classification.bat* : exécutable windows qui lance les scripts de classification (*2_model_selection.py* et *3_train_test_best_model.py*)
	* **word2vec** : méthodes de type word2vec (cbow et skip-gram)  
		* *0_word2vec_cbow_feature_extraction.ipynb* :  
		* *0_word2vec_skip_gram_feature_extraction.ipynb* :  
		* *1_word2vec_cbow_classification.ipynb* : classification cbow
		* *1_word2vec_skip_gram_classification.ipynb* : classification skip-gram  
		* *1_word2vec_glove.ipynb* : classification glove
	* Dans chacun de ces sous-dossiers :
		* *2_model_selection.py* : script qui implémente la sélection des modèles par k-fold cross-validation et learning curves
		* *3_train_test_best_model.py* : script qui entraîne et évalue le meilleur modèle sélectionné suite à la sélection de modèles

## Améliorations possibles à venir
* Elargir le type de blogs possibles (ex : skyblog, overblog, e-monsite, eklablog, etc.)
* Prétraitements restants : dans les datasets de documents, enlever les commentaires d'articles, enlever la description de l'auteur, enlever les références bibliographiques en notes de bas de page
* Implémenter les méthodes de classification du type réseaux de neurones (ex : transfomers, etc.)
* Version mise en prod de la partie réduction de dimension et visualisation
* Version mise en prod de la partie classification word2vec
* Pour tfidf, retourner pour chaque message les termes qui ont les scores les plus élevés
* Comprendre en détails l'implémentation de tfidf de sklearn