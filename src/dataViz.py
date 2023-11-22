# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 01:04:09 2023

@author: Diane Lantran
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from scipy.stats import pearsonr


# Import
FILE_PATH1 = "data/genres_v2.csv"
df = pd.read_csv(FILE_PATH1, sep=',')
colNames = df.columns.tolist()


#FONCTIONS QUI GENERENT LES GRAPHES

def energyDanceability():
    grouped_data = df.groupby('energy')['danceability'].mean().reset_index()
    plt.figure(figsize=(12, 8))
    plt.scatter(grouped_data['energy'], grouped_data['danceability'])
    plt.title('Danceabilité moyenne par energie')
    plt.xlabel("energy")
    plt.ylabel("danceability")
    plt.xticks(rotation=45, ha='right')
    correlation_coefficient, _ = pearsonr(grouped_data['energy'], grouped_data['danceability'])
    plt.text(0.8, 0.1, f"Correlation Coefficient: {correlation_coefficient:.2f}", 
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    plt.savefig('graphs/danceabiliteEnergie.png')
    plt.show()

def tempoEnergy():
    grouped_data = df.groupby('tempo')['energy'].median().reset_index()
    plt.figure(figsize=(12, 8))
    plt.scatter(grouped_data['tempo'], grouped_data['energy'])
    plt.title('Energie moyenne par valeur de tempo')
    plt.xlabel("tempo")
    plt.ylabel("energy")
    plt.xticks(rotation=45, ha='right')
    # Coefficient extrement proche de 0
    correlation_coefficient, _ = pearsonr(grouped_data['tempo'], grouped_data['energy'])
    plt.text(0.8, 0.1, f"Correlation Coefficient: {correlation_coefficient:.2f}", 
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    plt.savefig('graphs/tempoEnergie.png')
    plt.show()

def energyValence():
    grouped_data = df.groupby('energy')['valence'].mean().reset_index()
    plt.figure(figsize=(12, 8))
    plt.scatter(grouped_data['energy'], grouped_data['valence'])
    plt.title('Valence moyenne par energie')
    plt.xlabel("energy")
    plt.ylabel("valence")
    plt.xticks(rotation=45, ha='right')
    correlation_coefficient, _ = pearsonr(grouped_data['energy'], grouped_data['valence'])
    plt.text(0.8, 0.1, f"Correlation Coefficient: {correlation_coefficient:.2f}", 
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    plt.savefig('graphs/energieValence.png')
    plt.show()

def keyMode():
    grouped_data = df.groupby('key')['mode'].value_counts(normalize=True).unstack().fillna(0)
    majorKey = grouped_data[0]
    minorKey = grouped_data[1]
    plt.figure(figsize=(12, 8))
    plt.hist([majorKey, minorKey], bins=20, alpha=0.7, label=['Majeur', 'Mineur'])
    plt.title('Représentativité des modes')
    plt.xlabel("clé")
    plt.ylabel("mode, orange : majeur, bleu : mineur")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('graphs/cleMode.png')
    plt.show()

#diagramme moustache repartition valence par mode
def boxValMode():
    valMaj = df['valence'] #avec mode 0
    valMin = df['valence'] #avec mode 1
    data = [valMaj, valMin]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Répartition de la valence par mode')
    ax1.boxplot(data, showfliers=False)
    plt.savefig('graphs/ValModeRep.png')
    plt.show()

def instrumentalnessMoyTempo():
    bins = pd.cut(df['tempo'], bins=range(0, int(df['tempo'].max()) + 11, 10)) #regroupe les valeurs de tempo par tranche de 10

    # Group the data by the bins
    grouped_data = df.groupby(bins)['instrumentalness'].mean().reset_index()
    plt.figure(figsize=(12, 8))
    plt.bar(grouped_data.index.astype(str), grouped_data, color='red', label="Instumentalisation moyenne par tranche de tempo")
    plt.title('Instumentalisation moyenne par tranche de tempo')
    plt.xlabel("tempo")
    plt.ylabel("instrumentalness")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('graphs/meanInstruTempo.png')
    plt.show()

def repInstruSpeechMode():
    #a faire
    return 