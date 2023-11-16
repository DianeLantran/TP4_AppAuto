# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 01:04:09 2023

@author: Diane Lantran
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import numpy as np
import dataViz as dv
import seaborn as sns

# Import
FILE_PATH1 = "data/unpopular_songs.csv"
DATASET1 = pd.read_csv(FILE_PATH1, sep=',')

description1 = DATASET1.describe()
description1.to_markdown('description_songs.md')

FILE_PATH2 = "data/z_genre_of_artists.csv"
DATASET2 = pd.read_csv(FILE_PATH2, sep=',')

description2 = DATASET2.describe()
description2.to_markdown('description_genre.md')

# ajout des fonctions pour générer les graphs