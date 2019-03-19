import pandas as pd
import numpy as np
import progressbar as pbar
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from langid.langid import LanguageIdentifier, model


def get_indices_from_text_len(lyrics):
    print("Detecting outliers based on length...")
    lyric_len = np.empty(shape=(len(lyrics), 2))
    for i in pbar.progressbar(range(len(lyrics))):
        lyric_len[i, 1] = len(lyrics[i])
        lyric_len[i, 0] = i
    # remove any song with lyrics length that is too long (e.g. invalid lyrics)  based on z-score >= |3|
    lyric_len = StandardScaler().fit_transform(lyric_len)
    remove_indices = np.argwhere(abs(lyric_len[:, 1]) >= 3).ravel()
    print("Outliers detected based on length: {}".format(len(remove_indices)))
    return remove_indices


def clean_lyric_text(lyrics):
    pattern = "(\[|\{).{1,50}(\]|\})"
    empty_str = ''
    punctuations = string.punctuation.replace("'", empty_str)
    for i in pbar.progressbar(range(len(lyrics))):
        lyrics[i] = lyrics[i].lower()
        # clear tags & leading/trailing spaces
        lyrics[i] = re.sub(pattern, empty_str, lyrics[i]).strip()
        # clear punctuations (i.e. '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')
        lyrics[i] = lyrics[i].translate(
            str.maketrans(empty_str, empty_str, punctuations)
        )


def get_indices_from_tags(lyrics):
    pattern = "(\[|\{).{1,50}(\]|\})"
    remove_indices = []
    print("Detecting outliers based on tags...")
    for i in pbar.progressbar(range(len(lyrics))):
        if re.match(pattern, lyrics[i]) == None:
            remove_indices.append(i)
    print("Outliers detected based on tags: {}".format(len(remove_indices)))
    return remove_indices


def get_indices_from_word_count(lyrics, limits=(5, 1000)):
    remove_indices = []
    print("Detecting outliers based on {} < word count < {}...".format(
        limits[0], limits[1]))
    for i in pbar.progressbar(range(len(lyrics))):
        words = lyrics[i].split(' ')
        if len(words) > limits[1] or len(words) < limits[0]:
            remove_indices.append(i)
    print("Outliers detected based on word count: {}".format(len(remove_indices)))
    return remove_indices


def get_indices_from_unique_words(lyrics, unique_count_thresh=10):
    remove_indices = []
    print("Detecting outliers based on unique word count < {}...".format(
        unique_count_thresh))
    for i in pbar.progressbar(range(len(lyrics))):
        words = lyrics[i].split(' ')
        s = set()
        count = 0
        for w in words:
            if w not in s:
                count += 1
                s.add(w)
        if count < unique_count_thresh:
            remove_indices.append(i)
    print("Outliers detected based on unique word count: {}".format(
        len(remove_indices)))
    return remove_indices


def get_indices_from_char(lyrics):
    print("Detecting outliers based on characters...")
    pattern = '^[a-zA-Z0-9-\'\s]*$'
    remove_indices = []
    for i in pbar.progressbar(range(len(lyrics))):
        if re.match(pattern, lyrics[i]) == None:
            remove_indices.append(i)
    print("Outliers detected based on characters: {}".format(
        len(remove_indices)))
    return remove_indices


def get_indices_from_lang(lyrics, elligible_char_len=4, neg_prob_thresh=0.8, pos_thresh=0.8, lang='en', elligible_word_thresh=10):
    '''
    Returns indices of entries to remove based on language. Removal criteria is 
    based on percentage positive matches of elligible words in the lyric. A word in the 
    lyric is considered elligible if it meets the elligible_char_len specified.
    Since language detectors are not very good at identifying language based on 
    individual words, a word is only considered a negative match if it meets the 
    neg_prob_thresh specified.
    '''
    print("Detecting outliers based on language...")
    remove_indices = []
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    for i in pbar.progressbar(range(len(lyrics))):
        words = lyrics[i].split(' ')
        positives = 0
        negatives = 0
        unknowns = 0
        for w in words:
            if len(w.strip()) >= elligible_char_len:
                l, p = identifier.classify(w)
                if l != lang and p > neg_prob_thresh:
                    negatives += 1
                else:
                    positives += 1
        if positives + negatives == 0 or positives + negatives < elligible_word_thresh:
            remove_indices.append(i)
        elif positives / (positives + negatives) < pos_thresh:
            remove_indices.append(i)

    print("Outliers detected based on language: {}".format(len(remove_indices)))
    return remove_indices


def tfidf_vectorize(lyrics, y, ngram_range, min_df, token_pattern=r'\b\w+\b'):
    '''
    Uses TF-IDF vectorizer to convert lyric data to numeric features.
    Returns resulting data frame.
    '''
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=token_pattern,
    )
    X = vectorizer.fit_transform(lyrics)
    X = pd.DataFrame(X.toarray())
    cols = [i for i in range(X.shape[0])]
    cols.append('y')
    return pd.concat([X, y], axis=1), vectorizer


def get_class_based_data(df, class_id):
    '''
    Random generate equally sized dataset for binary classifiers of specified class
    by limiting the generated dataset size to minimum across all classes in dataset.
    '''
    size_limit = df['y'].value_counts().min()
    class_df = df.loc[df['y'] == class_id]
    non_class_df = df.loc[df['y'] != class_id]
    max_size = df['y'].value_counts().min()
    class_df = class_df.sample(n=max_size)
    non_class_df = non_class_df.sample(n=max_size)
    return pd.concat([class_df, non_class_df])
