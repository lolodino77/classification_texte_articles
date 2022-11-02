import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
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
pd.set_option("display.precision", 2)


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
    # On recuperer la classe minoritaire et son nombre de representants (son support)
    class_len = data[class_col_name].value_counts()
    name_of_minority_class = class_len.index[class_len.argmin()]
    number_of_minority_class = class_len[class_len.argmin()]

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


def do_cross_validation(X_train, y_train, scoring, num_iter, k):
    """Equilibre un dataset binaire non equilibre : il aura le meme nombre d'exemples de chaque classe

    Parametres: 
    X_train (numpy ndarray) : Les parametres 
    y_train (numpy ndarray) : Les etiquettes 
    scoring (string) : Le nom de la colonne du dataframe qui contient les classes 
                                Exemples : 
                                ['accuracy', 'precision', 'recall', 'f1', 'f1_macro'] 
                                ['f1_macro', 'f1_micro']
    num_iter (int) : Nombre d'iterations de la k-fold cross validation
    k (int) : Nombre de decoupages du train durant chaque etape de la k-fold cross validation 
                Exemple : k=10 en general
    """
    # Cross validation
    #Methode version automatisee facile grace a la fonction RepeatedStratifiedKFold de sklearn
    #Selection de modeles avec la k cross validation pour determiner le meilleur

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('RandomForest', RandomForestClassifier()))
    # models.append(('MLPClassifier', MLPClassifier(max_iter=500))) car diverge donc trop long
    models.append(('SGDClassifier', SGDClassifier()))
    models.append(('SVM', SVC()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=num_iter, random_state=None)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        for i, scores in cv_results.items():
            cv_results[i] = round(np.mean(scores), 4) #on fait la moyenne de chaque score (rappel, precision, etc.) pour les k experiences
        print((str(list(cv_results.items())[2:])+" ({0})").format(name)) #2: pour ignorer les info inutiles


def get_confusion_matrix(y_test, y_pred, model):
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


# Learning curves du modele selectionne : performances du modele en fonction de la taille du trainset
#Entrees
    #train_sizes (liste de float) : tailles du train en pourcentage 
    #cv_param : parametres de type kfold pour la cross validation
def get_learning_curve(model, X_train, y_train, cv_param, scoring, train_sizes, n_jobs=-1):
    # print("train_sizes =", 100 * train_sizes * len(y_train))
    train_sizes, train_scores, cv_scores = learning_curve(model, X_train, y_train, cv=cv_param, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    # learning_curve(AdaBoostClassifier(), X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)
    model_name = str(model)
    plt.figure()
    train_plot_label = scoring.capitalize() + " sur le trainset"
    cv_plot_label = scoring.capitalize() + " sur le cvset"
    title = scoring.capitalize() + " sur le trainset et sur le cvset en fonction de la taille du trainset pour " + model_name
    plt.plot(train_sizes, train_scores_mean, label=train_plot_label, color="b")
    plt.plot(train_sizes, cv_scores_mean, label=cv_plot_label, color="r")
    plt.title(title)
    plt.xlabel("Taille du trainset", fontsize=12)
    plt.ylabel(scoring.capitalize(), fontsize=12)
    plt.legend(loc="upper right")
    plt.show()


def plot_roc(y_true, y_pred):
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
    plt.show()