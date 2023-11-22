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
FILE_PATH1 = "data/genres_v2.csv"
df = pd.read_csv(FILE_PATH1, sep=',')

description = df.describe()
description.to_markdown('description_songs.md')


## corrélation des données entre elles
dv.energyDanceability()
dv.keyMode()
dv.energyValence()
dv.boxDurationMode()
dv.instrumentalnessMoyTempo()
dv.evolutionMoyDureeParInstru()
dv.evolutionMoyDureeParSpeech()
dv.moyInstruParAcous()