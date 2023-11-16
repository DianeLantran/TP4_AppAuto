import pandas as pd
import preprocessing as prep
import clustering

# Importation des bases de donnée
FILE_PATH = "../data/unpopular_songs.csv"
df = pd.read_csv(FILE_PATH, sep=',')


# Nettoyage des données : 


# Preprocessing
features = df.columns.difference(["track_name", "track_id"])
categorical_cols = ['track_artist', "explicit"]

# Preprocessing
df = prep.preprocess(df, categorical_cols, features)

