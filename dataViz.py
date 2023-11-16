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


# Import
df1 = pd.read_csv("data/unpopular_songs.csv", sep=',')
colNames1 = df1.columns.tolist()

df2 = pd.read_csv("data/z_genre_of_artists.csv", sep=',')
colNames2 = df2.columns.tolist()

# fonctions pour générer les graphs