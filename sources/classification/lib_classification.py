import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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


