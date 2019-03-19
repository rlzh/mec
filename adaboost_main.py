import pandas as pd
import numpy as np
from shared import preproc
from shared import const
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.manifold import Isomap
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)

n_comps = 1000
n_neighs = 50
embedding = Isomap(n_components=n_comps, n_neighbors=n_neighs)

parameters = {"n_estimators": [i for i in range(100, 1000, 100)]}


for i in spotify_df.y.unique():
    print("Training classifier for Class ({})...".format(i))
    # get data set for class i
    df = preproc.get_class_based_data(
        spotify_df,
        i,
        random_state=np.random.seed(0),
    )
    # preprocessing
    vectorized, vectorizer = preproc.tfidf_vectorize(
        df.lyrics.values,
        df.y.values,
        (1, 1),
        1,
    )
    # split features and labels
    X = vectorized.iloc[:, 0:vectorized.shape[1]-1]
    y = vectorized.y

    print(y.value_counts())

    # dimensionality reduction
    X = embedding.fit_transform(X.values)

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    clf_gscv = GridSearchCV(clf, param_grid=parameters, cv=10)
    clf_gscv.fit(X, y.values.ravel())
    print("Best estimator: {}".format(clf_gscv.best_estimator_))
    print("Best estimator: {}".format(clf_gscv.best_params_))
    print("Best score: {}".format(clf_gscv.best_score_))
    print()
