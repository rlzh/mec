import sys
import os
import re
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from nltk.corpus import stopwords
import nltk

#from pylab import *

import random

#from shared import preproc

df = pd.read_csv('./data/clean/clean_uneven_spotify.csv')
df = df.filter(['lyrics','y'], axis=1)
df.lyrics = df.lyrics.str.replace(r'\d+\s', '')

stopwords = stopwords.words('english')
# add_list = ["i'd", "i've", "i'll", "can't", "i'm", "what", "what's"]
# stopwords.append(add_list)
#lemma = nltk.wordnet.WordNetLemmatizer()
#print(lemma.lemmatize('leaves'))

def get_class_based_data(df, class_id, random_state=None):
    '''
    Random generate equally sized dataset for binary classifiers of specified class
    by limiting the generated dataset size to minimum across all classes in dataset.
    '''
    class_df = df.loc[df['y'] == class_id]
    non_class_df = df.loc[df['y'] != class_id]
    max_size = df['y'].value_counts().min()
    #max_size = len(class_df.y)
    class_df = class_df.sample(n=max_size)
    non_class_df = non_class_df.sample(n=max_size)
    if random_state != None:
        class_df = class_df.sample(
            n=max_size,
            random_state=random_state
        )
        non_class_df = non_class_df.sample(
            n=max_size,
            random_state=random_state
        )
    class_df.y = np.full(class_df.y.shape, 1)
    non_class_df.y = np.full(non_class_df.y.shape, -1)
    result = pd.concat([class_df, non_class_df])
    result.columns = df.columns
    result = result.sample(frac=1)
    # result['lyrics'] = result['lyrics'].str.split()
    # #result['lyrics'] = result['lyrics'].apply(lambda x: [lemma.lemmatize(item) for item in x])
    # result['lyrics'] = result['lyrics'].apply(lambda x: [item for item in x if item not in stopwords])
    # result['lyrics'] = result['lyrics'].str.join(' ')
    return result

def get_salient_phrases(clf, vectorizer, n=10):
    feature_importances_ = clf.feature_importances_
    feature_importance_ordering = np.argsort(feature_importances_)[::-1]
    feature_names = np.array(vectorizer.get_feature_names())
    top_n = feature_names[feature_importance_ordering][:n]
    return np.array(top_n)

def grid_search(df, y, name, vectorizer, lsa, estimators, param_grid,
                random_state=None, save_=True, verbose=1, cv=10, params_to_plot=[],
                scoring="accuracy"):

    best_estimators = []
    # t = time.time()
    
    print()
    print("Grid search using {}...".format(name))
    print("Param grid: {}".format(param_grid))
    print("Vectorizer = {}".format(vectorizer))
    # print("Even = {}".format(even))
    print()

    # input_df = utils.get_vectorized_df(input_df, vectorizer)
    # print(input_df.shape)
   

    # t0 = time.time()
    # get dataset for current class
    # df = utils.get_class_based_data(
    #     input_df,
    #     i,
    #     random_state=random_state,
    #     include_other_classes=True,
    #     limit_size=False,
    #     even_distrib=False,
    # )
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
        #random_state = 0
    )

    grid_result = gscv.fit(df.lyrics.values, y)
    # grid_result = gscv.fit(
    #     df.iloc[:, 0:df.shape[1]-1].values, df.y.values.ravel())
    feature_size = len(gscv.best_estimator_.named_steps['vectorizer'].get_feature_names())
    print("Feature size: {}".format(feature_size))
    print("Best: {} using {}".format(gscv.best_score_, gscv.best_params_))
    # print top 5 
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    results = np.array(list(zip(means, stds, params)))
    results = results[np.argsort(results[:, 0])[::-1]]
    top_results = []
    print("Best 5:")
    for j in range(5):
        result_str = "{}({}) with: {}".format(
            results[j, 0],
            results[j, 1],
            results[j, 2]
        )
        top_results.append(result_str)
        print(result_str)

    best_estimators.append(gscv.best_estimator_)
    return gscv.best_score_, gscv.best_params_

###############################################################################
#  Define tf-idf, LSA and SVM
###############################################################################

vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    ngram_range=(1, 1),
    min_df=2,
    # max_features=100,
    max_df=1.0,
    use_idf=True,
)

#svd = TruncatedSVD(n_components=50)
#lsa = make_pipeline(TruncatedSVD(n_components=30), Normalizer(copy=False))
lsa = TruncatedSVD(n_components=30)

clf = LinearSVC(C=1.0)

estimators = [
    ('vectorizer', vectorizer),
    ('lsa', lsa),
    ('clf', clf),
]
param_grid = {
        'vectorizer__min_df': [3, 5, 10],
        'vectorizer__max_df': [0.5, .25, 0.1],
        # 'vectorizer__ngram_range': [(1,1)],
        # 'vectorizer__max_features': [2500],
        # 'vectorizer__stop_words': [None, stop_words],
        # 'lsa__n_components': [10, 30, 50, 100, 150],
        # 'lsa__n_components': [10, 30, 50, 100],
        'lsa__n_components': [10, 30, 50],
        'clf__C': [1, 10, 100, 1000],
        # 'clf__base_estimator__max_depth':[1, 2],
}

###############################################################################
#  Run classification of the test articles
###############################################################################

def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

sum_list = [0, 0, 0, 0]
sum_auc = 0
sum_test = 0

c_list = []
p_list = []
test_list = []

h_df = get_class_based_data(df,1)
a_df = get_class_based_data(df,2)
s_df = get_class_based_data(df,3)
r_df = get_class_based_data(df,4)

y_Happy = h_df['y'].values
y_Angry = a_df['y'].values
y_Sad = s_df['y'].values
y_Relaxed = r_df['y'].values

# raw_set = (h_df, a_df, s_df, r_df)

h_df.drop(columns=['y'], inplace = True)
a_df.drop(columns=['y'], inplace = True)
s_df.drop(columns=['y'], inplace = True)
r_df.drop(columns=['y'], inplace = True)

# h_test = h_df.drop(columns=['y'])
# a_test = a_df.drop(columns=['y'])
# s_test = s_df.drop(columns=['y'])
# r_test = r_df.drop(columns=['y'])

x_set = (h_df, a_df, s_df, r_df)
y_set = (y_Happy, y_Angry, y_Sad, y_Relaxed)
y_name_set = ('Happy','Angry','Sad','Relaxed')

for x,y,y_name in zip(x_set, y_set, y_name_set):

    #test = df['x_angry_train'].values
    best_score, best_param = grid_search(
            x,
            y,
            y_name,
            vectorizer,
            lsa,
            estimators,
            param_grid,
            # random_state=random_state,
            # save_=save_,
            # verbose=verbose,
            # cv=cv,
            # params_to_plot=params_to_plot,
        )
    print("best score: for", y_name, "is: %f" % best_score) 
    #print("best param for", y_name, "is: ", best_param)
    #print(type(best_score))

    # c = best_param['clf__C']
    # n_components = best_param['lsa__n_components']
    # max_df = best_param['vectorizer__max_df']
    # min_df = best_param['vectorizer__min_df']

    # print(c, n_components, max_df, min_df)

    # vectorizer2 = TfidfVectorizer(
    #     stop_words=stopwords,
    #     ngram_range=(1, 1),
    #     min_df=min_df,
    #     # max_features=100,
    #     max_df=max_df,
    #     use_idf=True,
    # )

    # #svd = TruncatedSVD(n_components=50)
    # #lsa = make_pipeline(TruncatedSVD(n_components=30), Normalizer(copy=False))
    # svd2 = TruncatedSVD(n_components=n_components)
    # lsa2 = make_pipeline(svd2, Normalizer(copy=False))
    # #lsa2 = TruncatedSVD(n_components=n_components)

    # clf2 = LinearSVC(C=c)

    # df_tfidf = vectorizer2.fit_transform(x.lyrics.values)
    # print('TF-IDF output shape: ', df_tfidf.shape)

    # print("\nCross_val_score for", y_name, "is: %f" % scores.mean())

    # print("AUC Score for (LSA + SVM): %f" % metrics.accuracy_score(y_test, lsa_p))
    sum_auc += best_score
    # sum_test += scores.mean()
    # print("ROC Score for (LSA + SVM): %f" % metrics.roc_auc_score(y_test, lsa_p))
    #sum_roc += metrics.roc_auc_score(y_test, lsa_p)
    c_list.append(best_score)
    p_list.append(best_param)
    # test_list.append(scores.mean())

print(c_list)
print(p_list)
print(test_list)
avg_auc = float(sum_auc)/4
#avg_roc = float(sum_roc)/4
print("\nAverage score: %f" % avg_auc)
# avg_test = float(sum_test)/4
# print("\nAverage test score: %f" % avg_test)
#print("Average roc score: %f" % avg_roc)
#print(c_list)
# sum_result += avg_auc
# sum_list = [x + y for x, y in zip(sum_list, c_list)]
