import pandas as pd
import preprocessing as prep
import clustering

# Importation des bases de donnée
FILE_PATH1 = "../data/genres_v2.csv"
df1 = pd.read_csv(FILE_PATH1, sep=',')

# Nettoyage des données : 


# Preprocessing
features = df1.columns.difference(["type", "id", "uri", "track_href", 
                                   "analysis_url", "song_name",
                                   "Unnamed: 0", "title", "time_signature",
                                   "duration_ms"])
categorical_cols = ["genre"]
# Preprocessing
df = prep.preprocess(df1, categorical_cols, features)

results = clustering.cluster(df)