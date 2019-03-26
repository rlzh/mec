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
    return stop_words


def get_word_counts(lyrics, print_=False):
    word_count = np.empty(shape=(len(lyrics),))
    for i in range(len(lyrics)):
        if type(lyrics[i]) is str:
            words = lyrics[i].split(' ')
            word_count[i] = len(words)
        else:
            word_count[i] = 0
    if print_ == True:
        print("Word count info")
        print("min word count: {} (id: {})".format(
            np.min(word_count), np.argmin(word_count)))
        print("max word count: {} (id: {})".format(
            np.max(word_count), np.argmax(word_count)))
        print("ave word count: {}".format(np.mean(word_count)))
        print()
    return word_count


def get_unique_counts(lyrics, print_=False):
    unique_count = np.empty(shape=(len(lyrics),))
    for i in range(len(lyrics)):
        if type(lyrics[i]) is str:
            words = lyrics[i].split(' ')
            s = set()
            for w in words:
                if w not in s:
                    s.add(w)
                    unique_count[i] += 1
        else:
            unique_count[i] = 0
    if print_ == True:
        print("Unique count info")
        print("min unique count: {} (id: {})".format(
            np.min(unique_count), np.argmin(unique_count)))
        print("max unique count: {} (id: {})".format(
            np.max(unique_count), np.argmax(unique_count)))
        print("ave unique count: {}".format(np.mean(unique_count)))
        print()
    return unique_count


def get_emotion_counts(df, print_=False):
    class_distrib = df['y'].value_counts()
    if print_ == True:
        print("Emotion info")
        print("valence range: [{},{}]".format(
            df.valence.min(), df.valence.max()))
        print("arousal range: [{},{}]".format(
            df.arousal.min(), df.arousal.max()))
        print(class_distrib)
        print()
    return class_distrib


def get_top_singular_ngrams(vectorized, vectorizer, n=10):
    feature_names = np.array(vectorizer.get_feature_names())
    vectorized_sorted = np.sort(vectorized.toarray().flatten())[::-1]
    vectorized_sorting = np.argsort(vectorized.toarray()).flatten()[::-1]
    top_n = feature_names[vectorized_sorting][:n]
    top_n_scores = vectorized_sorted[:n].reshape(-1, )
    # top_n_scores = np.add(top_n_scores, 1.0).reshape(-1,)
    top_n = dict(zip(top_n, top_n_scores))
    return top_n


def get_top_idf_ngrams(vectorized, vectorizer, n=10):
    feature_names = np.array(vectorizer.get_feature_names())
    vectorized_sorted = np.sort(vectorizer.idf_.flatten())[::-1]
    vectorized_sorting = np.argsort(vectorizer.idf_).flatten()[::-1]
    top_n = feature_names[vectorized_sorting][:n]
    top_n_scores = vectorized_sorted[:n].reshape(-1, )
    # top_n_scores = np.add(top_n_scores, 1.0).reshape(-1,)
    top_n = dict(zip(top_n, top_n_scores))
    return top_n


def get_top_total_ngrams(vectorized, vectorizer, n=10):
    feature_names = np.array(vectorizer.get_feature_names())
    vectorized_sorted = np.sort(vectorized.toarray().sum(axis=0))[::-1]
    vectorized_sorting = np.argsort(vectorized.toarray().sum(axis=0))[::-1]
    top_n = feature_names[vectorized_sorting][:n]
    top_n_scores = vectorized_sorted[:n].reshape(-1,)
    # top_n_scores = np.add(top_n_scores, 1.0).reshape(-1,)
    top_n = dict(zip(top_n, top_n_scores))
    return top_n


def get_class_based_data(df, class_id, random_state=None, include_other_classes=True, limit_size=True, even_distrib=True):
    '''
    Random generate equally sized dataset for binary classifiers of specified class
    by limiting the generated dataset size to minimum across all classes in dataset.
    '''
    max_size = df['y'].value_counts().min()
    pos_df = df.loc[df['y'] == class_id]
    neg_df = df.loc[df['y'] != class_id]

    if limit_size == False:
        # no limit so take max between pos & neg df
        max_size = min(len(neg_df), len(pos_df))

    if include_other_classes:
        if even_distrib:
            # even distribution of negative class data between negative classes
            required_size = int(max_size / 3)
            min_size = df['y'].value_counts().min()
            if required_size > min_size:
                # not all classes can meet required size - need to update max_size, required_size
                required_size = min_size

            max_size = required_size * 3
            neg_df = pd.DataFrame([], columns=df.columns)
            for i in df.y.unique():
                if i != class_id:
                    rows = df.loc[df['y'] == i].sample(
                        n=required_size,
                        random_state=random_state
                    )
                    neg_df = pd.concat(
                        [neg_df, rows],
                        ignore_index=True
                    )
        else:
            neg_df = neg_df.sample(n=max_size, random_state=random_state)

    pos_df = pos_df.sample(
        n=max_size,
        random_state=random_state
    )
    pos_df.y = np.full(pos_df.y.shape, 1)
    result = pos_df
    if include_other_classes:
        print(neg_df.y.value_counts())
        neg_df.y = np.full(neg_df.y.shape, -1)
        result = pd.concat([pos_df, neg_df])
    result.columns = df.columns
    return result
