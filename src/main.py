import pandas as pd
import preprocessing as prep
import clustering

# Importation des bases de donnée
FILE_PATH1 = "../data/unpopular_songs.csv"
df1 = pd.read_csv(FILE_PATH1, sep=',')

FILE_PATH2 = "../data/z_genre_of_artists.csv"
df2 = pd.read_csv(FILE_PATH2, sep=',')

# Nettoyage des données : 


# Preprocessing
features = df1.columns.difference(["track_name", "track_id"])
categorical_cols = ['track_artist', "explicit"]

# Preprocessing
df = prep.preprocess(df1, categorical_cols, features)

