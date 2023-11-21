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
FILE_PATH = "data/unpopular_songs.csv"
DATASET = pd.read_csv(FILE_PATH, sep=',')

description = DATASET.describe()
description.to_markdown('description_songs.md')


# ajout des fonctions pour générer les graphs