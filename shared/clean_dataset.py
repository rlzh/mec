import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pbar
import const
import sys
from preproc import get_indices_from_lang, get_indices_from_lang_whole
from preproc import get_indices_from_unique_words
from preproc import get_indices_from_word_count
from preproc import clean_lyric_text
from preproc import get_indices_from_tags
from preproc import get_indices_from_text_len_z_score
from preproc import get_indices_from_char
from preproc import get_indices_from_lsa_z_score
from preproc import get_indices_from_unique_count_z_score
from utils import str_to_bool


def check_dup(df, s):
    init = (df.shape[0])
    df = df.drop_duplicates(subset=['song', 'artist'])
    if df.shape[0] != init:
        print(init)
        print(df.shape[0])
        print(s)


def clean(csv_path, word_count_range=(50, 800), unique_words_range=(20, 350)):
    ''' Load dataset from csv_path & clean  '''
    # load dataset
    df = pd.read_csv(csv_path, dtype=object)
    print()
    print("Cleaning {}...".format(csv_path))
    print()
    print("Uncleaned shape: {}".format(df.shape))
    start_rows = df.shape[0]
    # print nan count
    print("Missing per col: \n{}".format(df.isna().sum()))
    print()
    # remove any row with nan value(s)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any duplicate songs
    rows = len(df)
    df = df.drop_duplicates(subset=['song', 'artist'], keep='first')
    print("Dulicates removed: {}".format(rows - len(df)))
    print()
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
        limits=word_count_range
    )
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove songs based with lyric unique word count < thresh
    remove_indices = get_indices_from_unique_words(
        df.lyrics.values,
        limits=unique_words_range
    )
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any song with characters that are not alphanumeric with '
    remove_indices = get_indices_from_char(df.lyrics.values)
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove any song that is not english
    remove_indices = get_indices_from_lang_whole(df.lyrics.values)
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Total removed: {}".format(start_rows - df.shape[0]))
    print("Cleaned shape: {}".format(df.shape))
    print()
    return df


def gen_labels(df, cross_over_val=0, thresh=0, class_size=-1, class_distrib={}):
    ''' Dataset labelling '''
    # create emotion class label (i.e. Y) based on arousal & valence
    y = np.empty(shape=(df.shape[0], 2))
    va_list = pd.concat([df.valence, df.arousal], axis=1).values
    # # valence and arousal range may not be in [-1,1] so use z-score to normalize here
    # va_list = StandardScaler().fit_transform(va_list)
    # Quad 1: V > cross_over_val & A >= cross_over_val -> Happy
    # Quad 2: V <= cross_over_val & A > cross_over_val -> Angry
    # Quad 3: V < cross_over_val & A <= cross_over_val -> Sad
    # Quad 4: V >= cross_over_val & A < cross_over_val -> Relaxed
    remove_indices = []
    print("Labelling songs...")
    for i in pbar.progressbar(range(va_list.shape[0])):
        v = float(va_list[i, 0])
        a = float(va_list[i, 1])

        if v > cross_over_val and a >= cross_over_val:
            y[i, 0] = 1
        elif v <= cross_over_val and a > cross_over_val:
            y[i, 0] = 2
        elif v < cross_over_val and a <= cross_over_val:
            y[i, 0] = 3
        elif v >= cross_over_val and a < cross_over_val:
            y[i, 0] = 4

        # remove if euclidean distance below threshold
        y[i, 1] = ((v - cross_over_val)**2 + (a - cross_over_val)**2)**.5
        if y[i, 1] < thresh:
            remove_indices.append(i)

    y_df = pd.DataFrame(y, columns=['y', 'dist'])
    df = pd.concat([df, y_df], axis=1)
    print("Finished labelling songs! {} songs removed".format(len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if class_size > 0:
        class_size = min(df.y.value_counts().max(), class_size)
        print("Filtering songs based on class size={} and distrib={}...".format(
            class_size, class_distrib))
        result = pd.DataFrame([], columns=df.columns)
        for i in df.y.unique():
            pos_df = df.loc[df['y'] == i]
            # sort songs by distance from origin (decsending)
            pos_df = pos_df.sort_values(by='dist', ascending=False)
            distrib = 1.0 if i not in class_distrib else class_distrib[i]
            required_size = int(class_size * distrib)
            if len(pos_df) > required_size:
                rows = pos_df.iloc[:required_size]
                result = pd.concat([result, rows], ignore_index=True)
            else:
                result = pd.concat([result, pos_df], ignore_index=True)
        print("Done filtering songs! {} songs removed".format(
            len(df) - len(remove_indices)))
        df = result
    df = df.drop(['dist'], axis=1)
    # df = df.sample(frac=1)
    print("Class Distribution")
    print(df.y.value_counts())
    print("Labelled shape: {}".format(df.shape))
    print()
    return df


def main(*args):

    dry_run = False
    class_size = -1
    spotify_thresh = 0.2
    deezer_thresh = 0.8

    for arg in args:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if k == 'dry_run':
            dry_run = str_to_bool(v)
        elif k == 'class_size':
            class_size = int(v)
        elif k == 'spotify_thresh':
            spotify_thresh = float(v)
        elif k == 'deezer_thresh':
            deezer_thresh = float(v)

    print()
    print("--- Clean config ---")
    print("Dry run: {}".format(dry_run))
    print("Class size: {}".format(class_size))
    print("Spotify thresh: {}".format(spotify_thresh))
    print("Deezer thresh: {}".format(deezer_thresh))
    print("--------------------")
    print()

    # SPOTIFY DATASET
    cleaned_df = clean(const.GEN_SPOTIFY)
    df = gen_labels(
        cleaned_df,
        cross_over_val=0.5,
        thresh=spotify_thresh,
        class_size=class_size,
    )
    check_dup(df, "Error: Duplicates!!!!!!!!!!!!!!!!!")

    if dry_run == False:
        # save as csv (don't include indices in .csv)
        if class_size > 0:
            print("Saving result to {}...".format(const.CLEAN_SPOTIFY))
            df.to_csv(const.CLEAN_SPOTIFY, index=False)
        else:
            print("Saving result to {}...".format(const.CLEAN_UNEVEN_SPOTIFY))
            df.to_csv(const.CLEAN_UNEVEN_SPOTIFY, index=False)

    # DEEZER DATASET
    cleaned_df = clean(
        const.GEN_DEEZER,
        # word_count_range=(30, 600),
        # unique_words_range=(10, 300)
    )
    df = gen_labels(
        cleaned_df,
        cross_over_val=0,
        thresh=deezer_thresh,
        class_size=class_size,
    )
    check_dup(df, "Error: Duplicates!!!!!!!!!!!!!!!!!")
    if dry_run == False:
        # save as csv (don't include indices in .csv)
        if class_size > 0:
            print("Saving result to {}...".format(const.CLEAN_DEEZER))
            df.to_csv(const.CLEAN_DEEZER, index=False)
        else:
            print("Saving result to {}...".format(const.CLEAN_UNEVEN_DEEZER))
            df.to_csv(const.CLEAN_UNEVEN_DEEZER, index=False)


if __name__ == '__main__':
    main(*sys.argv[1:])
