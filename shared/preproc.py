import pandas as pd
import numpy as np
import progressbar as pbar
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from langid.langid import LanguageIdentifier, model
from langdetect import detect
from utils import get_word_counts
from utils import get_unique_counts


def get_indices_from_unique_count_z_score(lyrics):
    print("Detecting outliers based on unique count z-score >= |3|...")
    unique_count = get_unique_counts(lyrics)
    lyrics_scores = StandardScaler().fit_transform(unique_count)
    remove_indices = np.argwhere(abs(lyrics_scores[0]) >= 3).ravel()
    print("Outliers detected based on unique count z-score: {}".format(len(remove_indices)))
    return remove_indices


def get_indices_from_lsa_z_score(lyrics):
    print("Detecting outliers based on lsa z-score >= |3|...")
    vectorizer = TfidfVectorizer(
        max_df=0.5,
        max_features=10000,
        min_df=2,
        stop_words='english'
    )
    lyrics_tfidf = vectorizer.fit_transform(lyrics)
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    lyrics_lsa = lsa.fit_transform(lyrics_tfidf)
    lyrics_scores = StandardScaler().fit_transform(lyrics_lsa)
    remove_indices = np.argwhere(abs(lyrics_scores[0]) >= 3).ravel()
    print("Outliers detected based on lsa z-score: {}".format(len(remove_indices)))
    return remove_indices


def get_indices_from_text_len_z_score(lyrics):
    print("Detecting outliers based on length z-score >= |3|...")
    lyric_len = np.empty(shape=(len(lyrics), 2))
    for i in pbar.progressbar(range(len(lyrics))):
        lyric_len[i, 1] = len(lyrics[i])
        lyric_len[i, 0] = i
    # remove any song with lyrics length that is too long (e.g. invalid lyrics)  based on z-score >= |3|
    lyric_len = StandardScaler().fit_transform(lyric_len)
    remove_indices = np.argwhere(abs(lyric_len[:, 1]) >= 3).ravel()
    print("Outliers detected based on length z-score: {}".format(len(remove_indices)))
    return remove_indices


def clean_lyric_text(lyrics):
    pattern = "(\[.*?\])|(\dx)"
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
    pattern = "(\[.*?\])"
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
    word_counts = get_word_counts(lyrics)
    for i in pbar.progressbar(range(len(lyrics))):
        if word_counts[i] > limits[1] or word_counts[i] < limits[0]:
            remove_indices.append(i)
    print("Outliers detected based on word count: {}".format(len(remove_indices)))
    return remove_indices


def get_indices_from_unique_words(lyrics, limits=(10, 400)):
    remove_indices = []
    print("Detecting outliers based on unique {} < word count < {}...".format(
        limits[0], limits[1]))
    unique_counts = get_unique_counts(lyrics)
    for i in pbar.progressbar(range(len(lyrics))):
        if unique_counts[i] < limits[0] or unique_counts[i] > limits[1]:
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


def get_indices_from_lang_whole(lyrics):
    remove_indices = []
    print("Detecting outliers based on language...")
    for i in pbar.progressbar(range(len(lyrics))):
        if detect(lyrics[i]) != 'en':
            remove_indices.append(i)
    print("Outliers detected based on language: {}".format(len(remove_indices)))
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
