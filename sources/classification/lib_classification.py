import sys
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from pathlib import Path, PureWindowsPath
pd.set_option("display.precision", 2)
pd.set_option('display.max_colwidth', 40)
from lib_general import *
# sys.path.append(PureWindowsPath(r"C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_bapteme_philo\sources\create_dataset").as_posix())
sys.path.append("../..")

import sys
sys.path.append("../")

def get_merged_corpus_filenames(sys_argv):
    # python 2_model_selection.py command parquet corpus_edwardfeser_exapologist.parquet corpus_alexanderpruss_edwardfeser.parquet
    # python 2_model_selection.py ./data/input/ parquet
    # python 2_model_selection.py ./data/input/ csv
    # python 2_model_selection.py ./data/input/ all
    
    #argv[0] = le nom du fichier python execute
	files_to_open = sys_argv[1] # argument du script, si files_to_open==command execute le script sur les 
	# fichiers (datasets) entres en arguments dans la ligne de commande, 
	# mais si files_to_open!=command execute le script sur tous les fichiers du dossier ./data/input

	files_format = sys_argv[2] # format des fichiers datasets a ouvrir (parquet, csv, etc.), multiple si plusieurs formats
	print("len(sys_argv) =", len(sys_argv))
	# sert quand files_to_open==in_input_repertory, pour n'importer que les parquet, ou que les csv, etc.

	if(files_to_open == "command"):
		if(len(sys_argv) == 3): # cas quand il n'y a qu'un seul dataset => il faut creer une liste
			filenames = [sys_argv[3]]
		else: #cas quand il y a au moins deux datasets => pas besoin de creer de liste
			filenames = sys_argv[3:] # ignorer les 2 premiers arguments, le nom du script et files_to_open
	else:
		input_repertory = files_to_open.replace("/", "\\") # "/data/input/" ==> '\\data\\input\\'
		print("files in input : input_repertory =", input_repertory)
		filenames = glob.glob(os.path.join(input_repertory + "*." + files_format))
		filenames = [filename.split(input_repertory)[1] for filename in filenames] # enlever le path entier, garder que le nom du fichier

	return(filenames)


def get_merged_corpus_dataframe_from_filename(filename, format):
	if(format == "csv"):
		df = pd.read_csv("./data/input/merged_corpus/" + filename, encoding="utf-8")
	elif(format == "parquet"):
		df = pd.read_parquet("./data/input/merged_corpus/" + filename)
	return(df)


def get_balanced_binary_dataset(data, class_col_name):
    """Equilibre un dataset binaire non equilibre : il aura le meme nombre d'exemples de chaque classe

    Parametres: 
    data (pandas DataFrame) : Le dataframe pandas a deux classes (binaire)
                                Exemple : dataset_voitures
    class_col_name (string) : Le nom de la colonne du dataframe qui contient les classes 
                                Exemple : "categorie"

    Sortie:
    balanced_data (pandas DataFrame) : Le dataframe pandas equilibre
    """
    print("inside function get_balanced_binary_dataset")
    
    # On recuperer la classe minoritaire et son nombre de representants (son support)
    class_len = data[class_col_name].value_counts()
    name_of_minority_class = class_len.index[class_len.argmin()] # un entier binaire (0 ou 1)
    number_of_minority_class = class_len[name_of_minority_class] #name_of_minority_class car ici index = category 

    # On equilibre les classes
    minority_class = data.loc[data[class_col_name] == name_of_minority_class, :]
    majority_class = data.loc[data[class_col_name] != name_of_minority_class, :]
    majority_class_sampled = majority_class.sample(number_of_minority_class, random_state=42)
    balanced_data = pd.concat([majority_class_sampled, minority_class], ignore_index=True)

    # On melange les exemples et cree un id unique
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
    balanced_data.index = list(range(len(balanced_data)))
    balanced_data["id"] = balanced_data.index	#creation de l'id seulement apres equilibrage des classes et melange aleatoire

    return(balanced_data)


def get_train_and_test(data, features_col_names, class_col_name, id_col_name):
    """Separe les donnees en train et en test

    Parametres: 
    data (pandas DataFrame) : Le dataframe pandas a deux classes (binaire)
                                Exemple : dataset_voitures
    features_col_names (string) : Le nom de la colonne du dataframe qui contient les documents (messages) 
                                Exemple : "message_preprocessed"
    class_col_name (string) : Le nom de la colonne du dataframe qui contient les classes 
                                Exemple : "categorie"
    id_col_name (string) : Le nom de la colonne du dataframe qui contient les id uniques (cle primaire) 
                                Exemple : "categorie"


    Sortie:
    balanced_data (pandas DataFrame) : Le dataframe pandas equilibre
    """
    X = data[features_col_names]
    y = data[class_col_name]
    indices = data[id_col_name]
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.33, random_state=42)
    return(X_train, X_test, y_train, y_test, indices_train, indices_test)


def apply_tfidf_to_train(X_train):
    """Applique la transformation tfidf aux parametres du train
    (cree le vocabulaire a partir du train), implicitement transformation count puis tfidf 

    Parametres: 
    X_train (pandas DataFrame) : Les parametres du train

    Sorties:
    X_train_tfidf (pandas DataFrame) : Les parametres du train apres transformation tfidf
    tfidf_vectorizer (TfidfVectorizer) : La fonction tfidf (contenant entre autres le vocabulaire du train)                              
    """
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    return(X_train_tfidf, tfidf_vectorizer)


def apply_tfidf_to_test(X_test, tfidf_vectorizer):
    """Applique la transformation tfidf aux parametres du test en se basant sur le vocabulaire du train

    Parametres: 
    X_test (pandas DataFrame) : Les parametres du test  
    tfidf_vectorizer (TfidfVectorizer) : La fonction tfidf (contenant entre autres le vocabulaire du train)                              

    Sorties:
    X_test_tfidf (pandas DataFrame) : Les parametres du test apres transformation tfidf         
    """
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return(X_test_tfidf)


def apply_tfidf_to_train_and_test(X_train, X_test):
    """Applique la transformation tfidf aux parametres du train et du test 
    (en se basant sur le vocabulaire du train), implicitement transformation count puis tfidf 

    Parametres: 
    X_train (pandas DataFrame) : Les parametres du train
    X_test (pandas DataFrame) : Les parametres du test                                

    Sorties:
    X_train_tfidf (pandas DataFrame) : Les parametres du train apres transformation tfidf
    X_test_tfidf (pandas DataFrame) : Les parametres du test apres transformation tfidf         
    """
    X_train_tfidf, tfidf_vectorizer = apply_tfidf_to_train(X_train)
    X_test_tfidf = apply_tfidf_to_test(X_test, tfidf_vectorizer)

    return(X_train_tfidf, X_test_tfidf)


def do_cross_validation(X_train, y_train, scorings, num_iter, k, dataset_name=""):
    """Equilibre un dataset binaire non equilibre : il aura le meme nombre d'exemples de chaque classe

    Parametres: 
    X_train (numpy ndarray) : Les parametres 
    y_train (numpy ndarray int) : Les etiquettes au format int (le format string ne marche pas)
    scorings (liste de string) : Le nom des criteres (metriques) choisis pour evaluer le modele avec 
                                 les learning curves 
                                Exemples : 
                                ['accuracy', 'precision', 'recall', 'f1', 'f1_macro'] 
                                ['f1_macro', 'f1_micro']
    num_iter (int) : Nombre d'iterations de la k-fold cross validation
    k (int) : Nombre de decoupages du train durant chaque etape de la k-fold cross validation 
                Exemple : k=10 en general
    dataset_name (string) : Le nom du dataset pour creer un fichier de sortie avec le bon nom
    """
    # Cross validation
    #Methode version automatisee facile grace a la fonction RepeatedStratifiedKFold de sklearn
    #Selection de modeles avec la k cross validation pour determiner le meilleur

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('RandomForest', RandomForestClassifier()))
    # models.append(('MLPClassifier', MLPClassifier(max_iter=100))) car diverge donc trop long
    models.append(('SGDClassifier', SGDClassifier()))
    models.append(('SVM', SVC()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    
    # evaluate each model in turn
    print("models =", models)
    path = "./data/output/{}/cross_validation_results.txt".format(dataset_name)
    f = open(path, "w")

    for name, model in models:
        kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=num_iter, random_state=None)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scorings)
        for i, scores in cv_results.items():
            cv_results[i] = round(np.mean(scores), 4) #on fait la moyenne de chaque score (rappel, precision, etc.) pour les k experiences
        f.write((str(list(cv_results.items())[2:])+" ({0})\n").format(name))
        print((str(list(cv_results.items())[2:])+" ({0})").format(name)) #2: pour ignorer les info inutiles
    f.close()


def get_confusion_matrix(y_test, y_pred, model, savefig=True, dataset_name="", plotfig=True):
    """Affiche la matrice de confusion

    Parametres: 
    y_test (numpy ndarray) : Les etiquettes au format int (le format string ne marche pas)
    y_pred (numpy ndarray) : Les parametres du test  
    tfidf_vectorizer (TfidfVectorizer) : La fonction tfidf (contenant entre autres le vocabulaire du train)                              
    """    
    # Matrice de confusion
    false_label = "0 : Bapteme"
    true_label = "1 : Philosophie"
    confusion_matrix_var = confusion_matrix(y_test, y_pred, labels=model.classes_)
    group_names = ["Vrais baptême", "Faux philosophie", "Faux baptême", "Vrais philosophie"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    confusion_matrix_var.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        confusion_matrix_var.flatten()/np.sum(confusion_matrix_var)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    font_size = 15 #19
    plt.rcParams["figure.figsize"] = (10, 7) #taille du graphique, taille du plot
    ax = sb.heatmap(confusion_matrix_var, 
                    annot=labels, fmt="", cmap='Blues',
                    annot_kws={"size": font_size}, 
                    cbar_kws={'label': 'Nombre de sessions'})
    ax.set_xticklabels([false_label, true_label], Fontsize=font_size + 3)
    ax.set_yticklabels([false_label, true_label], Fontsize=font_size + 3)
    ax.figure.axes[-1].yaxis.label.set_size(font_size + 1)
    ax.figure.axes[-1].tick_params(labelsize=font_size - 2) 
    plt.title("Matrice de confusion", fontsize = font_size + 5)
    # xlabel = 'Catégories prédites\n\n Exactitude (bien classés) = {:0.2f} % ; Inexactitude (mal classés) = {:0.2f} %\n Précision (bonnes prédictions de robots / qualité) = {:0.2f} %\n Rappel (nombre de robots détectés / quantité) = {:0.2f} %\n F1 (synthèse de précision + rappel) = {:0.2f} %'.format(accuracy, (100 - accuracy), precision, recall, f1_score)
    plt.xlabel("Catégories prédites", fontsize=font_size + 3)
    plt.ylabel("Catégories réelles", fontsize=font_size + 3)
    if(savefig):
        path = "./data/output/{}/confusion_matrix_{}_{}".format(dataset_name, model.__class__.__name__)
        plt.savefig(path)
    if(plotfig):
        plt.show()


#Entrees
    #train_sizes (liste de float) : tailles du train en pourcentage 
    # y_train (numpy ndarray int) : Les etiquettes au format int (le format string ne marche pas)
    #cv_param : parametres de type kfold pour la cross validation
def get_learning_curve(model, X_train, y_train, cv_param, scoring, train_sizes, n_jobs=-1, 
                        savefig=True, dataset_name="", plotfig=True):
    """Affiche la learning curve du modele selectionne selon un critere
       Learning curve = performances du modele (selon un critere) en fonction de la taille du trainset

    Parametres: 
    model (modele sklearn) : Le modele de classification
    X_train (numpy ndarray) : Les parametres 
    y_train (numpy ndarray int) : Les etiquettes au format int (le format string ne marche pas)
    cv_param (liste) : Les parametres de la k-fold cross validation
    scoring (string) : Le nom du critere (metrique) choisi pour evaluer le modele avec les learning curves
                                Exemples : 'accuracy', 'precision', 'recall', 'f1', 'f1_macro'
    train_sizes (liste de float) : La liste des tailles des train (en pourcentage du train total original)
                                   Exemple : [0.2, 0.5, 0.7] = 20 % du train, 50 % du train, 70 % du train
    n_jobs (int) : Le nombre de jobs
    savefig (string) : Indique si on enregistre la learning curve dans une image
    dataset_name (string) : Dossier des sorties du dataframe etudie
    plotfig (string) : Indique si on affiche la learning curve sur une interface
    """
    train_sizes, train_scores, cv_scores = learning_curve(model, X_train, y_train, cv=cv_param, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)
    model_name = str(model)
    plt.figure(figsize=(14, 9))
    train_plot_label = scoring.capitalize() + " sur le trainset"
    cv_plot_label = scoring.capitalize() + " sur le cvset"
    title = scoring.capitalize() + " sur le trainset et sur le cvset en fonction de la taille du trainset pour " + model_name
    plt.plot(train_sizes, train_scores_mean, label=train_plot_label, color="b")
    plt.plot(train_sizes, cv_scores_mean, label=cv_plot_label, color="r")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, size=19)
    plt.xlabel("Taille du trainset", fontsize=18)
    plt.ylabel(scoring.capitalize(), fontsize=18)
    plt.legend(loc="upper right", prop={'size': 16})
    if(savefig):
        path = "./data/output/{}/learning_curve_{}_{}".format(dataset_name, model.__class__.__name__, scoring)
        plt.savefig(path)
    if(plotfig):
        plt.show()


def get_all_learning_curves(model, X_train, y_train, cv_param, scorings, train_sizes, n_jobs=-1, 
                            savefig=False, dataset_name="", plotfig=False):
    """Affiche les learning curves du modele selectionne selon les critere choisis par l'utilisateur
       Learning curve = performances du modele (selon un critere) en fonction de la taille du trainset

    Parametres: 
    model (modele sklearn) : Le modele de classification
    X_train (numpy ndarray) : Les parametres 
    y_train (numpy ndarray int) : Les etiquettes au format int (le format string ne marche pas)
    cv_param (liste) : Les parametres de la k-fold cross validation
    scorings (liste de string) : Le nom des criteres (metriques) choisis pour evaluer le modele avec 
                                 les learning curves 
                                Exemples : 
                                ['accuracy', 'precision', 'recall', 'f1', 'f1_macro'] 
                                ['f1_macro', 'f1_micro']
    train_sizes (liste de float) : La liste des tailles des train (en pourcentage du train total original)
                                   Exemple : [0.2, 0.5, 0.7] = 20 % du train, 50 % du train, 70 % du train
    n_jobs (int) : Le nombre de jobs
    savefig (string) : Indique si on enregistre les learning curves dans une image
    dataset_name (string) : Dossier des sorties du dataframe etudie
    """
    for scoring in scorings:
        print("scoring = ", scoring)
        get_learning_curve(model, X_train, y_train, cv_param, scoring, train_sizes, n_jobs, 
                            savefig, dataset_name, plotfig)


def plot_roc(model, y_true, y_pred, savefig=True, dataset_name="", plotfig=True):
    """Affiche la courbe roc

    Parametres: 
    y_true (numpy ndarray) : Les etiquettes correctes 
    y_pred (numpy ndarray) : Les etiquettes devinees par le modele
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    if(savefig):
        path = "./data/output/{}/roc_{}_{}".format(dataset_name, model.__class__.__name__)
        plt.savefig(path)
    if(plotfig):
        plt.show()


def get_classification_report()


def select_models(filenames, format_input):
    """Lance la selection de modeles (cross validation et learning curves)

    Parametres: 
    filenames (liste des string) : Les fichiers des differents datasets sur lesquels on execute cette fonction 
    """
    # Initialisation des variables necessaires
    id_col_name = "id"
    features_col_names = "message_preprocessed" 
    # class_col_name = "category"
    class_col_name = "category_bin"
    savefig = True
    print("in select_models()")
    print("current dir = ", os.getcwd())

    for filename in filenames:
        # Recupere le nom du dataset grace au nom du fichier du dataset filename
        print("filename =", filename)
        corpus_name = filename.split(".")[0]

        # Importer le dataset puis equilibrer ses classes
        corpus = get_merged_corpus_dataframe_from_filename(filename, format_input)
        print(corpus["category_bin"].value_counts())
        corpus = get_balanced_binary_dataset(corpus, class_col_name)
        print(corpus["category_bin"].value_counts())

        # Verifier la presence ou non de doublons
        check_duplicates(corpus, id_col_name)

        # Creation du train et du test
        X_train, X_test, y_train, y_test, indices_train, indices_test = get_train_and_test(corpus, features_col_names, class_col_name, id_col_name)
        X_train_tfidf, X_test_tfidf = apply_tfidf_to_train_and_test(X_train, X_test)

        # Creation du dossier de sorties si besoin
        if(savefig):
            os.makedirs("./data/output/" + corpus_name, exist_ok=True)

        # Cross validation
        # scorings = ['accuracy', 'f1_macro', "recall", "precision"]
        # num_iter = 2 #nombre de repetitions de la k-fold cross validation entiere
        # k = 10 #k de la k-fold cross validation
        # do_cross_validation(X_train_tfidf, y_train, scorings, num_iter, k, corpus_name)

        ## Learning curves (du meilleur modele)
        k = 10
        kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=20, random_state=None)
        cv_param = kfold
        num_experiences = 4
        train_sizes = np.linspace(0.2, 1.0, num_experiences)
        # n_jobs = -1
        model = SVC()

        scorings = ['accuracy', 'f1_macro', 'recall', 'precision']
        get_all_learning_curves(model, X_train_tfidf, y_train, cv_param, scorings, train_sizes, n_jobs=-1, 
                                    savefig=savefig, dataset_name=corpus_name)
        
        # get_all_learning_curves(model, X_train, y_train, cv_param, scorings, train_sizes, n_jobs=-1, 
        #                     savefig=False, dataset_name="", plotfig=False)