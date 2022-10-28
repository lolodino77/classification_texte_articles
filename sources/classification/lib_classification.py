import numpy as np
import pandas as pd

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

