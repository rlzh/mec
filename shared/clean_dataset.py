import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pbar
import const
from sklearn.preprocessing import StandardScaler
from langdetect import detect


def clean(csv_path):
    ''' Load dataset from csv_path & remove outliers '''
    # load dataset
    df = pd.read_csv(csv_path, dtype=object)
    start_rows = df.shape[0]
    print()
    print("Cleaning {}...".format(csv_path))
    # print nan count
    print("Missing per col: \n{}".format(df.isna().sum()))
    # remove any row with nan value(s)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # gather data on lyric length to filter invalid lyrics
    lyric_len = np.empty(shape=(df.shape[0], 2))
    print("Detecting outliers based on length...")
    for i in pbar.progressbar(range(len(df.lyrics))):
        lyric_len[i, 1] = len(df.lyrics[i])
        lyric_len[i, 0] = i
    # remove any song with lyrics length that is too long (e.g. invalid lyrics)  based on z-score >= |3|
    scaler = StandardScaler()
    lyric_len = scaler.fit_transform(lyric_len)
    remove_indices = np.argwhere(abs(lyric_len[:, 1]) >= 3).ravel()
    print("Outliers detected based on length: {}".format(len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any song with lyrics that do not have special lyric tags (e.g. [VERSE 1], [CHORUS], etc.)
    print("Detecting outliers based on tags...")
    pattern = "(\[|\{).{1,50}(\]|\})"
    empty_str = ''
    # {punctuations} - {'} (single quote)
    punctuations = string.punctuation.replace("'", empty_str)
    remove_indices = []
    for i in pbar.progressbar(range(len(df.lyrics))):
        df.lyrics[i] = df.lyrics[i].lower()
        # if re.match(pattern, df.lyrics[i]) == None:
        #     # print("{} by {}".format(df.song[i], df.artist[i]))
        #     remove_indices.append(i)
        # else:
        # clear tags & leading/trailing spaces
        df.lyrics[i] = re.sub(pattern, empty_str, df.lyrics[i]).strip()
        if len(df.lyrics[i]) == 0:
            remove_indices.append(i)
            continue
        # clear punctuations (i.e. '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')
        df.lyrics[i] = df.lyrics[i].translate(
            str.maketrans(empty_str, empty_str, punctuations)
        )
    print("Outliers detected based on tags: {}".format(len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove songs based with lyric word count >= limit
    space = ' '
    limit = 1000
    print("Detecting outliers based on word count >= {}...".format(limit))
    remove_indices = []
    for i in pbar.progressbar(range(len(df.lyrics))):
        words = df.lyrics[i].split(space)
        if len(words) >= limit:
            remove_indices.append(i)
    print("Outliers detected based on word count: {}".format(len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove songs based with lyric unique word count < thresh
    thresh = 10
    print("Detecting outliers based on unique word count < {}...".format(thresh))
    remove_indices = []
    for i in pbar.progressbar(range(len(df.lyrics))):
        words = df.lyrics[i].split(space)
        s = set()
        count = 0
        for w in words:
            if w not in s:
                count += 1
                s.add(w)
        if count < thresh:
            remove_indices.append(i)
    print("Outliers detected based on unique word count: {}".format(
        len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any song with lyrics that is not English based on langdetect package
    print("Detecting outliers based on language...")
    remove_indices.clear()
    for i in pbar.progressbar(range(len(df.lyrics))):
        try:
            lang = detect(df.lyrics[i])
            if lang != "en":
                remove_indices.append(i)
        except:
            print("Error during lang detect!")
            remove_indices.append(i)
    print("Outliers detected based on language: {}".format(len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Total outliers removed: {}".format(start_rows - df.shape[0]))
    print(df.shape)
    print()
    return df


def gen_labels(df, csv_path):
    ''' Dataset proprocessing & save to csv_path'''
    # create emotion class label (i.e. Y) based on arousal & valence
    y = np.empty(shape=(df.shape[0],))
    va_list = pd.concat([df.valence, df.arousal], axis=1).values
    # valence and arousal range may not be in [-1,1] so use z-score to normalize here
    va_list = StandardScaler().fit_transform(va_list)
    # Quad 1: V > 0 & A >=0 -> Happy
    # Quad 2: V <= 0 & A > 0 -> Angry
    # Quad 3: V < 0 & A <= 0 -> Sad
    # Quad 4: V >= 0 & A < 0 -> Relaxed
    for i in pbar.progressbar(range(va_list.shape[0])):
        v = float(va_list[i, 0])
        a = float(va_list[i, 1])
        if v > 0 and a >= 0:
            y[i] = 1
        elif v <= 0 and a > 0:
            y[i] = 2
        elif v < 0 and a <= 0:
            y[i] = 3
        elif v >= 0 and a < 0:
            y[i] = 4
    y_df = pd.DataFrame(y, columns=['y'])
    df = pd.concat([df, y_df], axis=1)

    # save as csv (don't include indices in .csv)
    df.to_csv(csv_path, index=False)
    return


# SPOTIFY DATASET
df = clean(const.GEN_SPOTIFY)
gen_labels(df, const.CLEAN_SPOTIFY)

# DEEZER DATASET
df = clean(const.GEN_DEEZER)
gen_labels(df, const.CLEAN_DEEZER)
