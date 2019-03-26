import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pbar
import const
from preproc import get_indices_from_lang, get_indices_from_lang_whole
from preproc import get_indices_from_unique_words
from preproc import get_indices_from_word_count
from preproc import clean_lyric_text
from preproc import get_indices_from_tags
from preproc import get_indices_from_text_len_z_score
from preproc import get_indices_from_char
from preproc import get_indices_from_lsa_z_score
from preproc import get_indices_from_unique_count_z_score


def clean(csv_path, word_count_range=(50, 800), unique_words_range=(20, 350)):
    ''' Load dataset from csv_path & remove outliers '''
    # load dataset
    df = pd.read_csv(csv_path, dtype=object)
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

    print("Total removed: {}".format(start_rows - df.shape[0]))
    print("Cleaned shape: {}".format(df.shape))
    print()
    return df


def gen_labels(df, csv_path, cross_over_val=0, thresh=0, class_size=-1, class_distrib=None):
    ''' Dataset proprocessing & save to csv_path'''
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
        remove = y[i, 1] < thresh
        if remove:
            remove_indices.append(i)

    y_df = pd.DataFrame(y, columns=['y', 'dist'])
    df = pd.concat([df, y_df], axis=1)
    print("{} songs removed".format(len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if class_size > 0:
        class_size = min(df.y.value_counts().max(), class_size)
        result = pd.DataFrame([], columns=df.columns)
        for i in df.y.unique():
            pos_df = df.loc[df.y == i]
            pos_df = pos_df.sort_values(by='dist', ascending=False)
            distrib = 1 if i not in class_distrib else class_distrib[i]
            required_size = int(class_size * distrib)
            if len(pos_df) > required_size:
                rows = pos_df.iloc[:required_size]
                result = pd.concat([result, rows], ignore_index=True)
            else:
                result = pd.concat([result, pos_df], ignore_index=True)
        print("{} songs removed".format(len(df) - len(remove_indices)))
        df = result
    df = df.drop(['dist'], axis=1)
    df = df.sample(frac=1)
    print("class distrib")
    print(df.y.value_counts())
    print("df shape: {}".format(df.shape))
    print()

    # save as csv (don't include indices in .csv)
    df.to_csv(csv_path, index=False)
    return


# SPOTIFY DATASET
df = clean(const.GEN_SPOTIFY)
gen_labels(
    df,
    const.CLEAN_SPOTIFY,
    cross_over_val=0.5,
    # thresh=0.2,
    class_distrib={
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 1.0,
    },
    class_size=750,
)

# DEEZER DATASET
df = clean(
    const.GEN_DEEZER,
    # word_count_range=(30, 600),
    # unique_words_range=(10, 300)
)
offset = 0.1
gen_labels(
    df,
    const.CLEAN_DEEZER,
    cross_over_val=0,
    # thresh=0.75,
    class_distrib={
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 1.0,
    },
    class_size=750,
)
