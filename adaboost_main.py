import pandas as pd
from shared import preproc
from shared import const

# preprocessing
vectorized_lyrics, vectorizer = preproc.tfidf_vectorize(
    const.CLEAN_SPOTIFY,
    (1, 1),
    1,
)
print(vectorized_lyrics.shape)
print(vectorized_lyrics.head())

# todo: implement adaboost classifier
