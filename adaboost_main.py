import pandas as pd
import datetime
import numpy as np
import time
import sys
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


def grid_search(input_df, name, estimators, param_grid, even=False, random_state=None, save_=True):

    f = open("adaboost_log.log", "a")
    if save_:
        f.write("\n==== {} - {} {}====\n".format(str(datetime.datetime.now()),
                                                 name, input_df.shape))
    best_estimators = []
    t = time.time()
    print()
    print("Grid search for {}...".format(name))
    print("Param grid: {}".format(param_grid))
    print()
    for i in input_df.y.unique():
        print()
        print("Training classifier for Class {}...".format(i))
        t0 = time.time()
        # get dataset for current class
        df = utils.get_class_based_data(
            input_df,
            i,
            random_state=random_state,
            include_other_classes=True,
            limit_size=True,
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
        if save_:
            # log results
            f.write("Class: {}\n".format(i))
            f.write("Best params: {}\n".format(gscv.best_params_))
            f.write("Best score: {}\n".format(gscv.best_score_))
            f.write("Top {} n-grams based on tfidf score: {}\n".format(20, top_n))
            f.write("Time: {}\n\n".format(time.time()-t0))
            f.flush()
            # save model
            save_path = "{}/{}_{}.sav".format(
                const.ADABOOST_MODEL_DIR, name, i)
            dump(gscv.best_estimator_, save_path)

        best_estimators.append(gscv.best_estimator_)
    if save_:
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


def evaluate(train_df, test_df, dataset_name, random_state=None):
    models = load_models(dataset_name, train_df.y.unique())
    for i in train_df.y.unique():
        print("Evaluating model for Class {}...".format(i))
        # get dataset for current class
        df = utils.get_class_based_data(
            train_df,
            i,
            random_state=random_state,
            include_other_classes=True,
            limit_size=True,
        )
        model = models[i]
        # refit tfidf using train data
        model.named_steps['tfidf'].fit_transform(df.lyrics.values)
        df = utils.get_class_based_data(
            test_df,
            i,
            random_state=random_state,
            include_other_classes=True,
            limit_size=True,
        )
        score = model.score(df.lyrics.values, df.y.values.ravel())
        print(score)


def main(*args):

    save_ = const.SAVE_DEFAULT
    random_state = const.RANDOM_STATE_DEFAULT
    test_size = 0.1
    dataset = ' + '.join([const.SPOTIFY, const.DEEZER])
    mode = ' + '.join([const.GSCV_MODE, const.EVAL_MODE])
    even_distrib = const.EVEN_DISTRIB_DEFAULT

    for arg in args:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if k == 'save':
            save_ = utils.str_to_bool(v)
        elif k == 'random_state':
            random_state = int(v)
        elif k == 'test_size':
            test_size = float(v)
        elif k == 'dataset':
            dataset = v
        elif k == 'mode':
            mode = v
        elif k == 'even_distrib':
            even_distrib = utils.str_to_bool(v)

    print()
    print("-- AdaBoost Config --")
    print("Save: {}".format(save_))
    print("Even distribution dataset: {}".format(even_distrib))
    print("Random state: {}".format(random_state))
    print("Test size: {}".format(test_size))
    print("Dataset: {}".format(dataset))
    print("Mode: {}".format(mode))
    print("--------------------")
    print()

    # load data
    spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
    if even_distrib == False:
        spotify_df = pd.read_csv(const.CLEAN_UNEVEN_SPOTIFY)
    spotify_train_df, spotify_test_df = utils.get_distrib_split(
        spotify_df,
        random_state=random_state,
        test_size=test_size,
    )
    deezer_df = pd.read_csv(const.CLEAN_DEEZER)
    if even_distrib == False:
        spotify_df = pd.read_csv(const.CLEAN_UNEVEN_SPOTIFY)
    deezer_train_df, deezer_test_df = utils.get_distrib_split(
        deezer_df,
        random_state=random_state,
        test_size=test_size,
    )

    if const.GSCV_MODE in mode:
        estimators = [
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            # ('svd', TruncatedSVD()),
            # ('normalize', Normalizer(copy=False)),
            ('adaboost', AdaBoostClassifier(n_estimators=800)),
        ]
        # spotify grid search
        if const.SPOTIFY in dataset:
            param_grid = {
                'tfidf__min_df': [5, 10],
                'tfidf__max_df': [1.0, 0.5],
                # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'tfidf__max_features': [None, 500],
                # 'tfidf__stop_words': [stop_words, None],
                'adaboost__n_estimators': [100, 1000],
            }
            grid_search(
                spotify_train_df,
                const.SPOTIFY,
                estimators,
                param_grid,
                random_state=random_state,
                save_=save_
            )

        # deezer grid search
        if const.DEEZER in dataset:
            param_grid = {
                'tfidf__min_df': [1, 5],
                'tfidf__max_df': [1.0, 0.5],
                # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # 'tfidf__max_features': [10000, 500],
                'tfidf__stop_words': [stop_words],  # , None],
                'adaboost__n_estimators': [500, 1000, 2000],
            }
            grid_search(
                deezer_train_df,
                const.DEEZER,
                estimators,
                param_grid,
                random_state=random_state,
                save_=save_
            )

    if const.EVAL_MODE in mode:
        if const.SPOTIFY in dataset:
            # evaluation on test data
            evaluate(
                spotify_train_df,
                spotify_test_df,
                const.SPOTIFY,
                random_state=random_state,
            )
        if const.DEEZER in dataset:
            evaluate(
                deezer_train_df,
                deezer_test_df,
                const.DEEZER,
                random_state=random_state,
            )


if __name__ == '__main__':
    main(*sys.argv[1:])
