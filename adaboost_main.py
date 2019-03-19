import pandas as pd
from shared import preproc
from shared import const
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


# preprocessing
# vectorized_lyrics, vectorizer = preproc.tfidf_vectorize(
#     const.CLEAN_SPOTIFY,
#     (1, 1),
#     1,
# )
# print(vectorized_lyrics.shape)
# print(vectorized_lyrics.head())

spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)

n_estimators = 100
clf = AdaBoostClassifier(n_estimators=n_estimators)

for i in spotify_df.y.unique():
    # get data set for class i
    df = preproc.get_class_based_data(
        spotify_df,
        i
    )
    scores = cross_val_scor(clf, df[:, :df.shape[0]-2], df.y, cv=10)
    print("Mean score for class {}: {}".format(i, scores.mean()))
