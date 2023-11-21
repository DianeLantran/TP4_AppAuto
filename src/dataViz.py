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
df = pd.read_csv("data/unpopular_songs.csv", sep=',')
colNames = df.columns.tolist()


# fonctions pour générer les graphs