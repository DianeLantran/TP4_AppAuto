import pandas as pd
import preprocessing as prep
import classification
import evaluationUtils as eva

# Importation des bases de donnée
FILE_PATH1 = "data/unpopular_songs.csv"
df1 = pd.read_csv(FILE_PATH1, sep=',')

FILE_PATH2 = "../data/z_genre_of_artists.csv"
df2 = pd.read_csv(FILE_PATH2, sep=',')

# Nettoyage des données : 


# Preprocessing
features = df1.columns.difference(['Booking_ID', 'booking_status'])
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', "booking_status"]

# Preprocessing
X, y = prep.preprocess(df, categorical_cols, features, target)

# Lance la pipeline de la classification
trained_models, resultsList = classification.classify(X, y)

#eva.graphScores(resultsList)
eva.ROCAndAUC(resultsList)
# Lance la pipeline de la classification

#eva.graphScores(resultsList)