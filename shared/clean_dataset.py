import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pbar
import const
from preproc import get_indices_from_lang
from preproc import get_indices_from_unique_words
from preproc import get_indices_from_word_count
from preproc import clean_lyric_text
from preproc import get_indices_from_tags
from preproc import get_indices_from_text_len
from preproc import get_indices_from_char


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
    remove_indices = get_indices_from_text_len(df.lyrics.values)
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any song with lyrics that do not have special lyric tags (e.g. [VERSE 1], [CHORUS], etc.)
    # remove_indices = get_indices_from_tags(df.lyrics.values)
    # df.drop(index=remove_indices, inplace=True)
    # df.reset_index(drop=True, inplace=True)

    # remove tags and punctuations from lyrics
    clean_lyric_text(df.lyrics.values)

    # remove songs based with lyric min < word count < max
    remove_indices = get_indices_from_word_count(
        df.lyrics.values,
        limits=(5, 1000)
    )
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove songs based with lyric unique word count < thresh
    remove_indices = get_indices_from_unique_words(
        df.lyrics.values,
        unique_count_thresh=10
    )
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any song with characters that are not alphanumeric with '
    remove_indices = get_indices_from_char(df.lyrics.values)
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any song with lyrics that is not English based on langdetect package
    # remove_indices = get_indices_from_lang(
    #     df.lyrics.values,
    #     elligible_char_len=5,
    #     neg_prob_thresh=0.5,
    #     pos_thresh=0.9,
    #     elligible_word_thresh=10,
    # )
    # df.drop(index=remove_indices, inplace=True)
    # df.reset_index(drop=True, inplace=True)

    print("Total removed: {}".format(start_rows - df.shape[0]))
    print(df.shape)
    print()
    return df


def gen_labels(df, csv_path, cross_over_val=0):
    ''' Dataset proprocessing & save to csv_path'''
    # create emotion class label (i.e. Y) based on arousal & valence
    y = np.empty(shape=(df.shape[0],))
    va_list = pd.concat([df.valence, df.arousal], axis=1).values
    # # valence and arousal range may not be in [-1,1] so use z-score to normalize here
    # va_list = StandardScaler().fit_transform(va_list)
    # Quad 1: V > cross_over_val & A >= cross_over_val -> Happy
    # Quad 2: V <= cross_over_val & A > cross_over_val -> Angry
    # Quad 3: V < cross_over_val & A <= cross_over_val -> Sad
    # Quad 4: V >= cross_over_val & A < cross_over_val -> Relaxed
    for i in pbar.progressbar(range(va_list.shape[0])):
        v = float(va_list[i, 0])
        a = float(va_list[i, 1])
        if v > cross_over_val and a >= cross_over_val:
            y[i] = 1
        elif v <= cross_over_val and a > cross_over_val:
            y[i] = 2
        elif v < cross_over_val and a <= cross_over_val:
            y[i] = 3
        elif v >= cross_over_val and a < cross_over_val:
            y[i] = 4
    y_df = pd.DataFrame(y, columns=['y'])
    df = pd.concat([df, y_df], axis=1)

    # save as csv (don't include indices in .csv)
    df.to_csv(csv_path, index=False)
    return


# SPOTIFY DATASET
df = clean(const.GEN_SPOTIFY)
gen_labels(df, const.CLEAN_SPOTIFY, cross_over_val=0.5)

# DEEZER DATASET
df = clean(const.GEN_DEEZER)
gen_labels(df, const.CLEAN_DEEZER)
