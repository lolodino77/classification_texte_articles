import sys
import os
sys.path.insert(0, "..\..")
sys.path.insert(0, "../..")
print("(lib_classification.py : sys.path =", sys.path)
import glob
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.transforms as mtrans
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

def get_merged_corpus_filenames(argv):
    """ Retourne tous les noms de fichiers des corpus dont on a besoin
    a partir des arguments de commande pour lancer le script python 2_model_selection.py
    
    Entree :
        argv (list of str) : Un a deux arguments (hormis l'argument python)            
            argv[1] = input files : "all" or "corpus_name.csv" or "corpus_name.parquet"
            argv[2] = input files format : only if argv[1] == "all", equals "csv" or "parquet"

    Sortie :
        output (list of str) : Tous les fichiers de corpus demandes par la commande de terminal
    
    # Exemples de commandes de terminal :
        # python 2_model_selection.py corpus_edwardfeser_exapologist.parquet
        # python 2_model_selection.py corpus_edwardfeser_exapologist.csv
        # python 2_model_selection.py all : model selection on all files (csv and parquet)
        # python 2_model_selection.py all csv : model selection on all csv files
        # python 2_model_selection.py all parquet : model selection on all parquet files
    """

    input_files = argv[1]
    filenames = input_files
    output = [filenames]
    if(input_files == "all"):
        input_files_format = argv[2]
        input_repertory = ".\\data\\input\\merged_corpus\\" #\\ et pas / car os renvoie \\
        filenames = glob.glob(os.path.join(input_repertory + "*." + input_files_format))

        print("filenames =", filenames)
        output = [filenames, input_files_format]

        if(len(filenames) != 0): #si os windows => chemin windows
            print("windows")
            filenames = [filename.split(input_repertory)[1] for filename in filenames] # enlever le path entier, garder que le nom du fichier
            output = [filenames, input_files_format]
        else: #si os unix => chemin unix
            print("unix")
            input_repertory = "./data/input/merged_corpus/"   
            filenames = glob.glob(os.path.join(input_repertory + "*." + input_files_format))
            filenames = [filename.split(input_repertory)[1] for filename in filenames] # enlever le path entier, garder que le nom du fichier
            output = [filenames, input_files_format]

    return(output)


def get_merged_corpus_dataframe_from_filename(filename):
    """ Retourne un dataframe pandas cree a partir d'un fichier de corpus (.csv ou .parquet) """
    format = filename.split(".")[1]

    if(format == "csv"):
       df = pd.read_csv("./data/input/merged_corpus/" + filename, encoding="utf-8")
    elif(format == "parquet"):
        df = pd.read_parquet("./data/input/merged_corpus/" + filename)

    return(df)


def get_balanced_binary_dataset(data, class_col_name):
    """ Equilibre un dataset binaire non equilibre : il aura le meme nombre d'exemples de chaque classe

    Entrees : 
        data (pandas DataFrame) : Le dataframe pandas a deux classes (binaire)
                                    Exemple : dataset_voitures
        class_col_name (string) : Le nom de la colonne du dataframe qui contient les classes 
                                    Exemple : "categorie"

    Sortie :
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
    """ Separe les donnees en train et en test (reprend la fonction train_test_split de sklearn)

    Entrees : 
        data (pandas DataFrame) : Le dataframe pandas a deux classes (binaire)
                                    Exemple : dataset_voitures
        features_col_names (string) : Le nom de la colonne du dataframe qui contient les documents (messages) 
                                    Exemple : "message_preprocessed"
        class_col_name (string) : Le nom de la colonne du dataframe qui contient les classes 
                                    Exemple : "categorie"
        id_col_name (string) : Le nom de la colonne du dataframe qui contient les id uniques (cle primaire) 
                                    Exemple : "categorie"

    Sortie :
        balanced_data (pandas DataFrame) : Le dataframe pandas equilibre
    """
    X = data[features_col_names]
    y = data[class_col_name]
    indices = data[id_col_name]
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.33, random_state=42)
    
    return(X_train, X_test, y_train, y_test, indices_train, indices_test)


def apply_tfidf_to_train(X_train):
    """ Applique la transformation tfidf aux parametres du train
    (cree le vocabulaire a partir du train), implicitement transformation count puis tfidf 

    Entrees : 
        X_train (pandas DataFrame) : Les parametres du train

    Sorties:
        X_train_tfidf (pandas DataFrame) : Les parametres du train apres transformation tfidf
        tfidf_vectorizer (TfidfVectorizer) : La fonction tfidf (contenant entre autres le vocabulaire du train)                              
    """
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    return(X_train_tfidf, tfidf_vectorizer)


def apply_tfidf_to_test(X_test, tfidf_vectorizer):
    """ Applique la transformation tfidf aux parametres du test en se basant sur le vocabulaire du train

    Entrees : 
        X_test (pandas DataFrame) : Les parametres du test  
        tfidf_vectorizer (TfidfVectorizer) : La fonction tfidf (contenant entre autres le vocabulaire du train)                              

    Sorties :
        X_test_tfidf (pandas DataFrame) : Les parametres du test apres transformation tfidf         
    """
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return(X_test_tfidf)


def apply_tfidf_to_train_and_test(X_train, X_test):
    """ Applique la transformation tfidf aux parametres du train et du test 
    (en se basant sur le vocabulaire du train), implicitement transformation count puis tfidf 

    Entrees : 
        X_train (pandas DataFrame) : Les parametres du train
        X_test (pandas DataFrame) : Les parametres du test                                

    Sorties :
        X_train_tfidf (pandas DataFrame) : Les parametres du train apres transformation tfidf
        X_test_tfidf (pandas DataFrame) : Les parametres du test apres transformation tfidf         
    """
    X_train_tfidf, tfidf_vectorizer = apply_tfidf_to_train(X_train)
    X_test_tfidf = apply_tfidf_to_test(X_test, tfidf_vectorizer)

    return(X_train_tfidf, X_test_tfidf)


def save_cross_validation(X_train, y_train, scorings, num_iter, k, dataset_name, stage):
    """ Lance une k-fold cross validation et enregistre les resultats dans un fichier .txt

    Entrees : 
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
        stage (string) : L'etape a laquelle on fait la cross-validation (select_model pour selection de modeles
                            ou best_model si on lance sur un seul modele)
    """
    # Cross validation
    #Methode version automatisee facile grace a la fonction RepeatedStratifiedKFold de sklearn
    #Selection de modeles avec la k cross validation pour determiner le meilleur

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('RandomForest', RandomForestClassifier()))
    # models.append(('SGDClassifier', SGDClassifier()))
    # models.append(('SVM', SVC()))
    # models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    # models.append(('MLPClassifier', MLPClassifier(max_iter=100))) car diverge donc trop long
     
    # evaluate each model in turn
    print("models =", models)
    path = "./data/output/{}/{}/cross_validation_results.txt".format(dataset_name, stage)
    print("\n\nos.getcwd() =", os.getcwd())
    f = open(path, "w")

    for name, model in models:
        kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=num_iter, random_state=None)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scorings, n_jobs=-1)
        for i, scores in cv_results.items():
            cv_results[i] = round(np.mean(scores), 4) #on fait la moyenne de chaque score (rappel, precision, etc.) pour les k experiences
        f.write((str(list(cv_results.items())[2:])+" ({0})\n").format(name))
        print((str(list(cv_results.items())[2:])+" ({0})").format(name)) #2: pour ignorer les info inutiles
    f.close()


def make_output_dir(dataset_name):
    if not os.path.exists("./data/output/{}".format(dataset_name)):
        os.makedirs("./data/output/{}".format(dataset_name))
    if not os.path.exists("./data/output/{}/select_model".format(dataset_name)):
        os.makedirs("./data/output/{}/select_model".format(dataset_name))
    if not os.path.exists("./data/output/{}/best_model".format(dataset_name)):
        os.makedirs("./data/output/{}/best_model".format(dataset_name)) 


def save_confusion_matrix(y_test, y_pred, class_names, model, dataset_name):
    """ Cree et enregistre la matrice de confusion d'un modele au format .png 

    Entrees : 
        y_test (numpy ndarray) : Les etiquettes au format int (le format string ne marche pas)
        y_pred (numpy ndarray) : Les parametres du test
        class_names (dictionary) : {"0":class_zero_name_map, "1":class_one_name_map}
        tfidf_vectorizer (TfidfVectorizer) : La fonction tfidf (contenant entre autres le vocabulaire du train)                              
    """
    # Matrice de confusion
    class_zero_name = class_names["0"]
    class_one_name = class_names["1"]
    class_zero_name_map = "0 : {}".format(class_zero_name) #"0 : Philosophie"
    class_one_name_map = "1 : {}".format(class_one_name) #"1 : Philosophie"
    confusion_matrix_var = confusion_matrix(y_test, y_pred, labels=model.classes_)
    group_names = ["Vrais {}".format(class_zero_name), "Faux {}".format(class_one_name), 
                    "Faux {}".format(class_zero_name), "Vrais {}".format(class_one_name)]
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
    ax.set_xticklabels([class_zero_name_map, class_one_name_map], fontsize=font_size + 3)
    ax.set_yticklabels([class_zero_name_map, class_one_name_map], fontsize=font_size + 3)
    ax.figure.axes[-1].yaxis.label.set_size(font_size + 1)
    ax.figure.axes[-1].tick_params(labelsize=font_size - 2) 
    plt.title("Matrice de confusion", fontsize = font_size + 5)
    # xlabel = 'Catégories prédites\n\n Exactitude (bien classés) = {:0.2f} % ; Inexactitude (mal classés) = {:0.2f} %\n Précision (bonnes prédictions de robots / qualité) = {:0.2f} %\n Rappel (nombre de robots détectés / quantité) = {:0.2f} %\n F1 (synthèse de précision + rappel) = {:0.2f} %'.format(accuracy, (100 - accuracy), precision, recall, f1_score)
    plt.xlabel("Catégories prédites", fontsize=font_size + 3)
    plt.ylabel("Catégories réelles", fontsize=font_size + 3)
    print("dataset_name =", dataset_name)

    path = "./data/output/{}/best_model/confusion_matrix_{}".format(dataset_name, model.__class__.__name__)
    plt.savefig(path)


def save_learning_curves_multiple_models(models, X_train, y_train, cv_param, scoring, train_sizes, dataset_name):
    """ Enregistre au format .png plusieurs learning curves selon plusieurs metriques pour plusieurs modeles
    (Par exemple, sur un graph de learning curve selon l'accuracy, on aura les courbes de chaque modele) 
    
    Entrees :
        models (modele sklearn) : Les modeles de classification
        X_train (numpy ndarray) : Les parametres du trainset 
        y_train (numpy ndarray int) : Les etiquettes du trainset au format int (le format string ne marche pas)
        cv_param (liste) : Les parametres de la k-fold cross validation
        scoring (string) : Le nom du critere (metrique) choisi pour evaluer le modele avec les learning curves
                                    Exemples : 'accuracy', 'precision', 'recall', 'f1', 'f1_macro'
        train_sizes (liste de float) : La liste des tailles des train (en pourcentage du train total original)
                                    Exemple : [0.2, 0.5, 0.7] = 20 % du train, 50 % du train, 70 % du train
        dataset_name (string) : Le nom du dataset qui sera le dossier des fichiers de sorties
    """
    colors = iter(cm.rainbow(np.linspace(0, 1, len(models))))
    fig, ax = plt.subplots(figsize=(14, 9))
    linewidth = 3
    trans_y = -2

    zip_list = list(zip(models, colors))
    model_tuple, color = zip_list[0]
    name, model = model_tuple
    print("model =", name)
    print("shape X_train =", len(y_train))
    n_jobs = -1
    train_sizes, train_scores, cv_scores = learning_curve(model, X_train, y_train, cv=cv_param, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    model_name = str(model)
    ax.plot(train_sizes, train_scores_mean, label=model_name, color=color, linewidth=linewidth)
    ax.plot(train_sizes, cv_scores_mean, color=color, linewidth=linewidth)
    for i in range(1, len(zip_list)):
        model_tuple, color = zip_list[i]
        name, model = model_tuple
        train_sizes, train_scores, cv_scores = learning_curve(model, X_train, y_train, cv=cv_param, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        cv_scores_mean = np.mean(cv_scores, axis=1)
        model_name = str(model)
        title = scoring.capitalize() + " sur le trainset et sur le cvset \n en fonction de la taille du trainset"
        tr = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=-(trans_y*i), units='points')
        ax.plot(train_sizes, train_scores_mean, label=model_name, color=color, linewidth=linewidth, transform=tr)
        # ax.plot(train_sizes, train_scores_mean, label=model_name, color=color, linewidth=linewidth)
        ax.plot(train_sizes, cv_scores_mean, color=color, linewidth=linewidth)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(title, size=19)
        plt.xlabel("Taille du trainset", fontsize=18)
        plt.ylabel(scoring.capitalize(), fontsize=18)
        plt.legend(loc="best", prop={'size': 16})
    path = "./data/output/{}/select_model/learning_curve_{}".format(dataset_name, scoring)
    plt.savefig(path)


#Entrees
    #train_sizes (liste de float) : tailles du train en pourcentage 
    # y_train (numpy ndarray int) : Les etiquettes au format int (le format string ne marche pas)
    #cv_param : parametres de type kfold pour la cross validation
def save_learning_curve(model, X_train, y_train, cv_param, scoring, train_sizes, dataset_name, stage, n_jobs=-1):
    """Affiche la learning curve du modele selectionne selon un critere
       Learning curve = performances du modele (selon un critere) en fonction de la taille du trainset

    Entrees : 
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
        stage (string) : Etape a laquelle on se trouve, soit "select_model" (selection de modeles)
                            soit "best_model" (modele final garde apres a la fin de la selection de modeles)
    """
    train_sizes, train_scores, cv_scores = learning_curve(model, X_train, y_train, cv=cv_param, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    print("train_scores =", train_scores)
    print("cv_scores =", cv_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)
    model_name = str(model)
    plt.figure(figsize=(14, 9))
    train_plot_label = scoring.capitalize() + " sur le trainset"
    cv_plot_label = scoring.capitalize() + " sur le cvset"
    title = scoring.capitalize() + " sur le trainset et sur le cvset en fonction de la taille du trainset pour " + model_name
    print("train_sizes =", train_sizes)
    print("train_scores_mean =", train_scores_mean)
    print("cv_scores_mean =", cv_scores_mean)
    
    plt.plot(train_sizes, train_scores_mean, label=train_plot_label, color="b")
    plt.plot(train_sizes, cv_scores_mean, label=cv_plot_label, color="r")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, size=19)
    plt.xlabel("Taille du trainset", fontsize=18)
    plt.ylabel(scoring.capitalize(), fontsize=18)
    plt.legend(loc="upper right", prop={'size': 16})
    
    path = "./data/output/{}/{}/learning_curve_{}_{}".format(dataset_name, stage, model.__class__.__name__, scoring)
    plt.savefig(path)


def save_all_learning_curves(model, X_train, y_train, cv_param, scorings, train_sizes, dataset_name, stage, n_jobs=-1):
    """Sauvegarde en fichiers images les learning curves du modele selectionne selon les critere choisis 
    par l'utilisateur.
    Learning curve = performances du modele (selon un critere) en fonction de la taille du trainset

    Entrees : 
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
        dataset_name (string) : Dossier des sorties du dataframe etudie
    """
    for scoring in scorings:
        print("scoring =", scoring)
        save_learning_curve(model, X_train, y_train, cv_param, scoring, train_sizes, dataset_name, stage, n_jobs)


def save_roc(model, y_true, y_pred, dataset_name):
    """ Enregistre la courbe roc dans un fichier .png

    Entrees : 
        model (sklearn model) : Le modele supervise entraine 
        y_true (numpy ndarray) : Les etiquettes correctes 
        y_pred (numpy ndarray) : Les etiquettes devinees par le modele
        dataset_name (str) : Le nom du dataset, ex : chat_chien, pomme_poire
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
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("Receiver operating characteristic example", fontsize=19)
    plt.legend(loc="lower right", prop={'size': 16})

    path = "./data/output/{}/best_model/roc_{}".format(dataset_name, model.__class__.__name__)
    plt.savefig(path)


def save_classification_report(y_test, y_pred, dataset_name, model):
    """ Sauvegarde .txt le classification report de sklearn (metriques pour chaque classe) dans un fichier
    
    Entrees :
        y_test (numpy ndarray) : Les etiquettes au format int (le format string ne marche pas)
        y_pred (numpy ndarray) : Les parametres du test
        dataset_name (string) : Le nom du corpus
        model (sklearn model) : Le modele evalue
    """
    report = classification_report(y_test, y_pred)
    print("type report = ", type(report))
    print("report =", report)
    path = "./data/output/{}/best_model/classification_report_{}.txt".format(dataset_name, model.__class__.__name__)
    print("path =", path)
    f = open(path, "w")
    f.write(report)
    f.close()


def save_false_predictions(corpus, dataset_name, indices_test, y_test, y_pred, class_names):
    """ Enregistre dans un fichier .csv les exemples a propos desquels le modele s'est trompe 
    
    Entrees :
        corpus (pandas DataFrame) : Le corpus
        dataset_name (string) : Le nom du corpus
        indices_test (list of string or pandas Series of string) : Les id du testset 
        y_test (numpy ndarray) : Les etiquettes au format int (le format string ne marche pas)
        y_pred (numpy ndarray) : Les parametres du test
        class_names (dictionary) : Le dictionnaire qui fait la correspondance entre les noms des classes en string
                                    avec les noms des classes en int
                                    Exemple : {"0":chien, "1":chat}
    """
    corpus_test = pd.DataFrame({"id":corpus.iloc[indices_test].id, "message": corpus.iloc[indices_test].message, "truth":y_test, "pred":y_pred})
    corpus_test_errors = corpus_test.query("truth != pred")
    corpus_test_errors = corpus_test_errors[["id", "truth", "pred", "message"]]
    corpus_test_errors["truth"] = np.select([corpus_test_errors["truth"] == 0], [class_names["0"]], default=class_names["1"])
    corpus_test_errors["pred"] = np.select([corpus_test_errors["pred"] == 0], [class_names["0"]], default=class_names["1"])
    corpus_test_errors.to_csv("./data/output/{}/best_model/false_predictions.csv".format(dataset_name), index=False, header=True)


def save_model_diagnostics(corpus, X_train, y_train, y_test, y_pred, indices_test, class_names, model, 
                            dataset_name):
    """ Evalue les performances d'un modele et sauvegarde les resultats (dans des fichiers .txt et .png)
    
    Entrees :
        X_train (numpy ndarray) : Les parametres 
        y_train (numpy ndarray int) : Les etiquettes au format int (le format string ne marche pas)
        y_test (numpy ndarray) : Les etiquettes au format int (le format string ne marche pas)
        y_pred (numpy ndarray) : Les parametres du test
        indices_test (list of string or pandas Series of string) : Les id du testset 
        class_names (dictionary) : Le dictionnaire qui fait la correspondance entre les noms des classes en string
                                    avec les noms des classes en int
                                    Exemple : {"0":chien, "1":chat}
        model (sklearn model) : Le modele a evaluer
        dataset_name (string) : Le nom du dataset
    """
    # Classification report
    save_classification_report(y_test, y_pred, dataset_name, model)

    # Matrice de confusion
    save_confusion_matrix(y_test, y_pred, class_names, model, dataset_name)

    # Courbe ROC
    save_roc(model, y_test, y_pred, dataset_name)

    # Learning curves
    k = 10
    kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=3, random_state=None)
    cv_param = kfold
    num_experiences = 4
    train_sizes = np.linspace(0.2, 1.0, num_experiences)
    n_jobs = -1
    scorings = ['accuracy', 'f1_macro', 'recall', 'precision']
    save_all_learning_curves(model, X_train, y_train, cv_param, scorings, train_sizes, 
                                dataset_name, "best_model", n_jobs)

    # Fausses predictions
    save_false_predictions(corpus, dataset_name, indices_test, y_test, y_pred, class_names)


def select_models(corpus, corpus_name, id_col_name, class_col_name, features_col_names):
    """ Lance la selection de modeles par k-fold cross-validation et par learning curves sur plusieurs modeles
    
    Entrees :
        corpus (pandas DataFrame) : Le corpus
        corpus_name (string) : Le nom du corpus
        id_col_name (string) : Le nom de la colonne du dataframe qui contient les id uniques (cle primaire) 
                                Exemple : "categorie"
        class_col_name (string) : Le nom de la colonne du dataframe qui contient les classes 
                                Exemple : "categorie"
        features_col_names (string) : Le nom de la colonne du dataframe qui contient les documents (messages) 
                                Exemple : "message_preprocessed"
    """
    print(corpus["category_bin"].value_counts())
    corpus = get_balanced_binary_dataset(corpus, class_col_name)
    print(corpus["category_bin"].value_counts())

    # Verifier la presence ou non de doublons
    check_duplicates(corpus, id_col_name)

    # Creation du train et du test
    X_train, X_test, y_train, y_test, indices_train, indices_test = get_train_and_test(corpus, features_col_names, class_col_name, id_col_name)
    X_train_tfidf, X_test_tfidf = apply_tfidf_to_train_and_test(X_train, X_test)

    # K-fold cross validation
    scorings = ['accuracy', 'f1_macro', "recall", "precision"]
    num_iter = 2 #nombre de repetitions de la k-fold cross validation entiere
    k = 10 #k de la k-fold cross validation
    print("corpus_name =", corpus_name)
    save_cross_validation(X_train_tfidf, y_train, scorings, num_iter, k, corpus_name, stage="select_model")

    ## Learning curves (du meilleur modele)
    k = 10
    kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=2, random_state=None)
    cv_param = kfold
    num_experiences = 2
    train_sizes = np.linspace(0.2, 1.0, num_experiences)
    # n_jobs = -1
    models = []
    # models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models.append(('RandomForest', RandomForestClassifier()))
    # models.append(('SGDClassifier', SGDClassifier()))
    # models.append(('SVM', SVC()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

    # scorings = ['accuracy', 'f1_macro', 'recall', 'precision']
    # scorings = ['accuracy', 'f1_macro', 'recall']
    scorings = ["accuracy"]
    for scoring in scorings:
        save_learning_curves_multiple_models(models, X_train_tfidf, y_train, cv_param, scoring, train_sizes, 
                                            corpus_name)