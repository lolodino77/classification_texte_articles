import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px


def plot_word_vectors_3d(word_vec, words, n_kept):
    """Plot en 3D de certains mots

    Parametres: 
    word_vec (numpy ndarray) : Les vecteurs des mots a afficher
                                Exemple : dataset_voitures
    words (string) : Les mots 
                                Exemple : ["chien", "chat"]
    n_kept (string) : Le nombre de mots a afficher (car souvent impossible de tous les afficher sinon illisible)
    """
    n_kept = 150
    fig = px.scatter_3d(x=word_vec[0:n_kept,0], y=word_vec[0:n_kept,1], z=word_vec[0:n_kept,2],
                text=words[0:n_kept])
    fig.show()


def plot_word_vectors_2d(word_vec, words, n_kept):
    """Plot en 2D de certains mots

    Parametres: 
    word_vec (numpy ndarray) : Les vecteurs des mots a afficher
                                Exemple : dataset_voitures
    words (string) : Les mots 
                                Exemple : ["chien", "chat"]
    n_kept (string) : Le nombre de mots a afficher (car souvent impossible de tous les afficher sinon illisible)
    """
    plt.figure(figsize=(12, 7))
    sb.scatterplot(
        x=word_vec[0:n_kept,0], y=word_vec[0:n_kept,1],
        legend="auto",
        alpha=0.3,
        s=10
    )
    for i in range(n_kept):
        plt.text(x=word_vec[i,0], y=word_vec[i,1], s=words[i])

    plt.xlabel("tsne_2d_x", size=16)
    plt.ylabel("tsne_2d_y", size=16)
    plt.title("Plot des vecteurs mots en 2D", size=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()