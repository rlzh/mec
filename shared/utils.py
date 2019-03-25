import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
import nltk


def get_stop_words():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    f = open("stopwords.txt", 'r')
    for l in f:
        if len(l.strip()) > 0:
            stop_words.add(l.strip())


def get_word_counts(lyrics, print=False):
    word_count = np.empty(shape=(len(lyrics),))
    for i in range(len(lyrics)):
        words = lyrics[i].split(' ')
        word_count[i] = len(words)
    if print == True:
        print("Word count info")
        print("min word count: {} (id: {})".format(
            np.min(word_count), np.argmin(word_count)))
        print("max word count: {} (id: {})".format(
            np.max(word_count), np.argmax(word_count)))
        print("ave word count: {}".format(np.mean(word_count)))
        print()
    return word_count


def get_unique_counts(lyrics, print=False):
    unique_count = np.empty(shape=(len(lyrics),))
    for i in range(len(lyrics)):
        words = lyrics[i].split(' ')
        s = set()
        for w in words:
            if w not in s:
                s.add(w)
                unique_count[i] += 1
    if print == True:
        print("Unique count info")
        print("min unique count: {} (id: {})".format(
            np.min(unique_count), np.argmin(unique_count)))
        print("max unique count: {} (id: {})".format(
            np.max(unique_count), np.argmax(unique_count)))
        print("ave unique count: {}".format(np.mean(unique_count)))
        print()
    return unique_count


def get_emotion_counts(df, print=False):
    class_distrib = df['y'].value_counts()
    if print == True:
        print("Emotion info")
        print("valence range: [{},{}]".format(
            df.valence.min(), df.valence.max()))
        print("arousal range: [{},{}]".format(
            df.arousal.min(), df.arousal.max()))
        print(class_distrib)
        print()
    return class_distrib


def get_top_tfidf_ngrams(vectorized, vectorizer, n=10, scaled=False):
    feature_array = np.array(vectorizer.get_feature_names())
    vectorized_sorted = np.sort(vectorized.toarray().flatten())[::-1]
    vectorized_sorting = np.argsort(vectorized.toarray()).flatten()[::-1]
    top_n = feature_array[vectorized_sorting][:n]
    top_n_scores = vectorized_sorted[:n].reshape(-1, 1)
    if scaled == True:
        top_n_scores = MinMaxScaler().fit_transform(top_n_scores)
    top_n_scores = np.add(top_n_scores, 1.0).reshape(-1,)
    top_n = dict(zip(top_n, top_n_scores))
    return top_n


def get_class_based_data(df, class_id, random_state=None, include_other_classes=True, limit_size=True):
    '''
    Random generate equally sized dataset for binary classifiers of specified class
    by limiting the generated dataset size to minimum across all classes in dataset.
    '''
    max_size = df['y'].value_counts().min()
    class_df = df.loc[df['y'] == class_id]
    non_class_df = df.loc[df['y'] != class_id]
    if limit_size == False:
        max_size = min(len(non_class_df), len(class_df))

    class_df = class_df.sample(n=max_size)
    if include_other_classes:
        non_class_df = non_class_df.sample(n=max_size)

    if random_state != None:
        class_df = class_df.sample(
            n=max_size,
            random_state=random_state
        )
        if include_other_classes:
            non_class_df = non_class_df.sample(
                n=max_size,
                random_state=random_state
            )
    class_df.y = np.full(class_df.y.shape, 1)
    result = class_df
    if include_other_classes:
        non_class_df.y = np.full(non_class_df.y.shape, -1)
        result = pd.concat([class_df, non_class_df])
    result.columns = df.columns
    return result
