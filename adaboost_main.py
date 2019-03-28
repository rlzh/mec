import pandas as pd
import datetime
import numpy as np
import time
from shared import utils
from shared import const
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.externals.joblib import load, dump


# load stop words
stop_words = utils.get_stop_words()


def grid_search(input_df, name, estimators, param_grid, log=True):

    f = open("adaboost_log.log", "a")
    if log:
        f.write("\n==== {} - {} {}====\n".format(str(datetime.datetime.now()),
                                                 name, input_df.shape))
    best_estimators = []
    t = time.time()

    for i in input_df.y.unique():
        print("Training classifier for Class {}...".format(i))
        t0 = time.time()
        # get dataset for current class
        df = utils.get_class_based_data(
            input_df,
            i,
            random_state=np.random.seed(0),
            include_other_classes=True,
            limit_size=True,
            even_distrib=True,
        )
        pipe = Pipeline(estimators, memory=None)
        # run grid search
        gscv = GridSearchCV(
            pipe,
            cv=10,
            verbose=1,
            iid=False,
            param_grid=param_grid,
            # scoring='f1',
            n_jobs=3,
        )
        gscv.fit(df.lyrics.values, df.y.values.ravel())
        print("Best params: {}".format(gscv.best_params_))
        print("Best score: {}".format(gscv.best_score_))

        # get top n-grams
        tfidf = gscv.best_estimator_.named_steps['tfidf']
        pos_tfidf = tfidf.fit_transform(df.loc[df.y == 1].lyrics.values)
        top_n = utils.get_top_idf_ngrams(pos_tfidf, tfidf, n=20)
        print("Top n-grams: {}".format(top_n))
        # save results to file
        if log:
            f.write("Class: {}\n".format(i))
            f.write("Best params: {}\n".format(gscv.best_params_))
            f.write("Best score: {}\n".format(gscv.best_score_))
            f.write("Top {} n-grams based on tfidf score: {}\n".format(20, top_n))
            f.write("Time: {}\n\n".format(time.time()-t0))
            f.flush()
        best_estimators.append(gscv.best_estimator_)
        # save model
        save_path = "{}/{}_{}.sav".format(const.ADABOOST_MODEL_DIR, name, i)
        dump(gscv.best_estimator_, save_path)
    if log:
        f.write("==== Total time {} ====\n".format(time.time()-t))
    f.close()
    return best_estimators


def load_models(dataset_name, classes):
    models = {}
    for i in classes:
        save_path = "{}/{}_{}.sav".format(
            const.ADABOOST_MODEL_DIR, dataset_name, i)
        print("Loading model from {}...".format(save_path))
        model = load(save_path)
        models[i] = model
    return models


def evaluate(input_df, dataset_name):
    models = load_models(dataset_name, input_df.y.unique())
    for i in input_df.y.unique():
        print("Evaluating model for Class {}...".format(i))
        # get dataset for current class
        df = utils.get_class_based_data(
            input_df,
            i,
            random_state=np.random.seed(0),
            include_other_classes=True,
            limit_size=True,
            even_distrib=True,
        )
        model = models[i]
        score = model.score(df.lyrics.values, df.y.values.ravel())
        print(score)


spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
spotify_train_df, spotify_test_df = utils.get_even_distrib_split(
    spotify_df,
    random_state=0,
    test_size=0.1
)
deezer_df = pd.read_csv(const.CLEAN_DEEZER)
deezer_train_df, deezer_test_df = utils.get_even_distrib_split(
    deezer_df,
    random_state=0,
    test_size=0.1
)

estimators = [
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    # ('svd', TruncatedSVD()),
    # ('normalize', Normalizer(copy=False)),
    ('adaboost', AdaBoostClassifier(n_estimators=800)),
]
# spotify grid search
param_grid = {
    'tfidf__min_df': [1, 5],
    'tfidf__max_df': [1.0, 0.5],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__max_features': [10000, 1000],
    'tfidf__stop_words': [stop_words, None],
    'adaboost__n_estimators': [1000, 2000, 3000],
}
grid_search(spotify_train_df, const.SPOTIFY, estimators, param_grid, log=True)

# deezer grid search
param_grid = {
    'tfidf__min_df': [1, 5, 10],
    'tfidf__max_df': [1.0, 0.5],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__max_features': [10000, 500],
    'tfidf__stop_words': [stop_words, None],
    'adaboost__n_estimators': [500, 1000, 2000],
}
grid_search(deezer_train_df, const.DEEZER, estimators, param_grid, log=True)

# evaluate(deezer_df, const.DEEZER)
# evaluate(spotify_df, const.SPOTIFY)
# train_test_score(pd.concat((deezer_df, spotify_df)))
