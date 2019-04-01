import pandas as pd
import datetime
import numpy as np
import time
import pydotplus
import sys
import matplotlib.pyplot as plt
from shared import utils
from shared import const
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.externals.joblib import load, dump
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import call

# load stop words
stop_words = utils.get_stop_words()


def visualize_tree(estimator):
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("tree.png")
    # plt.imshow(Image(graph.create_png()))
    # plt.axis('off')
    # plt.show()


def grid_search(input_df, name, vectorizer, estimators, param_grid,
                random_state=None, save_=True, verbose=1, cv=10, params_to_plot=[],
                scoring="accuracy"):

    f = open("adaboost_log.log", "a")
    if save_:
        f.write("\n==== {} - {} {}====\n".format(str(datetime.datetime.now()),
                                                 name, input_df.shape))
    best_estimators = []
    t = time.time()
    print()
    print("Grid search using {}...".format(name))
    print("Param grid: {}".format(param_grid))
    print("Vectorizer = {}".format(vectorizer))
    print()

    # input_df = utils.get_vectorized_df(input_df, vectorizer)
    # print(input_df.shape)

    for i in [2]:  # input_df.y.unique():
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
            even_distrib=False,
        )
        pipe = Pipeline(estimators, memory=None)

        # run grid search
        gscv = GridSearchCV(
            pipe,
            cv=cv,
            verbose=verbose,
            iid=False,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=-1,
        )

        grid_result = gscv.fit(df.lyrics.values, df.y.values.ravel())
        # grid_result = gscv.fit(
        #     df.iloc[:, 0:df.shape[1]-1].values, df.y.values.ravel())
        print("Best: {} using {}".format(gscv.best_score_, gscv.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        # get top n-grams
        # pos_tfidf = tfidf.fit_transform(df.loc[df.y == 1].lyrics.values)
        # top_n = utils.get_top_idf_ngrams(pos_tfidf, tfidf, n=20)
        # print("Top n-grams: {}".format(top_n))
        # save results to file
        if save_:
            # log
            f.write("Class: {}\n".format(i))
            f.write("Best params: {}\n".format(gscv.best_params_))
            f.write("Best score: {}\n".format(gscv.best_score_))
            # f.write("Top {} n-grams based on tfidf score: {}\n".format(20, top_n))
            f.write("Time: {}\n\n".format(time.time()-t0))
            f.flush()
            # save model
            save_path = "{}/{}_{}.sav".format(
                const.ADABOOST_MODEL_DIR, name, i)
            dump(gscv.best_estimator_, save_path)

        best_estimators.append(gscv.best_estimator_)

        for param_name in params_to_plot:
            param_to_score = utils.get_scoring_to_param(
                grid_result,
                param_name
            )
            plt.figure()
            plt.plot(param_to_score[0, :], param_to_score[1, :])
            plt.title("{} vs {}".format(param_name, scoring))
            plt.xlabel(param_name)
            plt.ylabel(scoring)
            plt.draw()
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
    mode = ' + '.join([const.GSCV_MODE])
    even_distrib = const.EVEN_DISTRIB_DEFAULT
    verbose = 1
    cv = 10
    even_distrib_train = False

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
        elif k == 'verbose':
            verbose = int(v)
        elif k == 'cv':
            cv = int(v)
        elif k == 'even_distrib_train':
            even_distrib_train = utils.str_to_bool(v)
    print()
    print("-- AdaBoost Config --")
    print("save: {}".format(save_))
    print("even_distrib: {}".format(even_distrib))
    print("even_distrib_train: {}".format(even_distrib_train))
    print("random_state: {}".format(random_state))
    print("test_size: {}".format(test_size))
    print("dataset: {}".format(dataset))
    print("mode: {}".format(mode))
    print("verbose: {}".format(verbose))
    print("cv: {}".format(cv))
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
        stump = DecisionTreeClassifier(
            max_depth=1,
        )
        vectorizer = CountVectorizer(
            # stop_words=stop_words,
            ngram_range=(1, 1),
            min_df=2,
            # max_features=100,
            max_df=.9,
        )
        estimators = [
            ('vectorizer', CountVectorizer(stop_words=stop_words)),
            ('adaboost', AdaBoostClassifier(
                base_estimator=stump, random_state=random_state)),
        ]
        param_grid = {
            'vectorizer__min_df': [2, 3, 4, 5],
            'vectorizer__max_df': [.9, 0.7, 1.0],
            # 'vectorizer__max_features': [None],
            # 'vectorizer__stop_words': [None, stop_words],
            'adaboost__n_estimators': [4000],
            'adaboost__learning_rate': [0.7, 1.0],
        }
        params_to_plot = [
            # 'vectorizer__max_df',
            # 'vectorizer__min_df',
            # 'adaboost__n_estimators',
            # 'adaboost__learning_rate',
        ]
        # spotify grid search
        if const.SPOTIFY in dataset:

            best_estimators = grid_search(
                spotify_train_df,
                const.SPOTIFY,
                vectorizer,
                estimators,
                param_grid,
                random_state=random_state,
                save_=save_,
                verbose=verbose,
                cv=cv,
                params_to_plot=params_to_plot,
            )

            # test: create decision tree visualization image
            # adaboost = best_estimators[0].named_steps['adaboost']
            # visualize_tree(adaboost.estimators_[0])

        # deezer grid search
        if const.DEEZER in dataset:
            grid_search(
                deezer_train_df,
                const.DEEZER,
                vectorizer,
                estimators,
                param_grid,
                random_state=random_state,
                save_=save_,
                verbose=verbose,
                cv=cv,
                params_to_plot=params_to_plot,
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
    plt.show()


if __name__ == '__main__':
    main(*sys.argv[1:])
