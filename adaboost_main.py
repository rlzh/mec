import pandas as pd
from shared import preproc
from shared import const

# preprocessing
# vectorized_lyrics, vectorizer = preproc.tfidf_vectorize(
#     const.CLEAN_SPOTIFY,
#     (1, 1),
#     1,
# )
# print(vectorized_lyrics.shape)
# print(vectorized_lyrics.head())

spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)

for i in spotify_df.y.unique():
    class_df = preproc.get_class_based_data(
        spotify_df,
        i
    )


# todo: implement adaboost classifier
