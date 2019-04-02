import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pbar
import const
import sys
import math
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

    # remove any song with lyrics that do not have special lyric tags(e.g. [VERSE 1], [CHORUS], etc.)
    remove_indices = get_indices_from_tags(df.lyrics.values)
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

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

    # remove any song with characters that are not alphanumeric with
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


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def rotate_via_numpy(p, degrees):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = p[0], p[1]
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return float(m.T[0]), float(m.T[1])


def gen_labels(df, cross_over_val=0, thresh=0, data_size=-1, class_distrib={}, class_per_quadrant=1, dist_mode='furthest_from_origin'):
    ''' Dataset labelling '''
    # create emotion class label (i.e. Y) based on arousal & valence
    y = np.empty(shape=(df.shape[0], 2))

    va_list = pd.concat([df.valence, df.arousal], axis=1)
    va_list.columns = ['valence', 'arousal']
    va_list = va_list.astype(float)
    print("valence range: [{},{}]".format(
        va_list.valence.min(), va_list.valence.max()))
    print("arousal range: [{},{}]".format(
        va_list.arousal.min(), va_list.arousal.max()))
    reference_point = (cross_over_val, cross_over_val)
    if dist_mode == 'furthest_from_mean':
        reference_point = (va_list.valence.mean(), va_list.arousal.mean())
    print("mean valence-arousal: {}".format((va_list.valence.mean(),
                                             va_list.arousal.mean())))

    # # valence and arousal range may not be in [-1,1] so use z-score to normalize here
    # va_list = StandardScaler().fit_transform(va_list)
    # Quad 1: V > cross_over_val & A >= cross_over_val -> Happy
    # Quad 2: V <= cross_over_val & A > cross_over_val -> Angry
    # Quad 3: V < cross_over_val & A <= cross_over_val -> Sad
    # Quad 4: V >= cross_over_val & A < cross_over_val -> Relaxed
    remove_indices = []
    print("Labelling songs...")
    p0 = (va_list.valence.values.max(), 0)
    divisor = 90 / class_per_quadrant

    for i in pbar.progressbar(range(va_list.shape[0])):
        v = float(va_list.iloc[i, 0])
        a = float(va_list.iloc[i, 1])
        alpha = angle_between((v - cross_over_val, a - cross_over_val), p0)

        if dist_mode == 'closest_to_extremes':
            if v > cross_over_val and a >= cross_over_val:
                reference_point = (va_list.valence.max(),
                                   va_list.arousal.max())
            elif v <= cross_over_val and a > cross_over_val:
                reference_point = (va_list.valence.min(),
                                   va_list.arousal.max())
            elif v < cross_over_val and a <= cross_over_val:
                reference_point = (va_list.valence.min(),
                                   va_list.arousal.min())
            elif v >= cross_over_val and a < cross_over_val:
                reference_point = (va_list.valence.max(),
                                   va_list.arousal.min())

        # determine class label based on angle between axis
        y[i, 0] = int(alpha / divisor) + 1

        # remove if euclidean distance below threshold
        y[i, 1] = ((v - reference_point[0])**2 +
                   (a - reference_point[1])**2)**.5
        if dist_mode == 'closest_to_extremes':
            if y[i, 1] > thresh:
                remove_indices.append(i)
        else:
            if y[i, 1] < thresh:
                remove_indices.append(i)

    y_df = pd.DataFrame(y, columns=['y', 'dist'])
    df = pd.concat([df, y_df], axis=1)
    ascending = True if dist_mode == 'closest_to_extremes' else False
    print("Finished labelling songs! {} songs removed".format(len(remove_indices)))
    df.drop(index=remove_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if dist_mode == 'furthest_from_mean':
        result = pd.DataFrame([], columns=df.columns)
        df = df.sort_values(by='dist', ascending=False)
        df.reset_index(drop=True, inplace=True)
        df = df.iloc[:data_size]
        df.reset_index(drop=True, inplace=True)
    elif data_size > 0:
        data_size = min(df.y.value_counts().max(), data_size)
        print("Filtering songs based on class size={} and distrib={}...".format(
            data_size, class_distrib))
        result = pd.DataFrame([], columns=df.columns)
        for i in df.y.unique():
            pos_df = df.loc[df['y'] == i]
            # sort songs by distance from origin (decsending)
            pos_df = pos_df.sort_values(by='dist', ascending=ascending)
            distrib = 1.0 if i not in class_distrib else class_distrib[i]
            required_size = int(data_size * distrib)
            if len(pos_df) > required_size:
                rows = pos_df.iloc[:required_size]
                result = pd.concat([result, rows], ignore_index=True)
            else:
                result = pd.concat([result, pos_df], ignore_index=True)
        print("Done filtering songs! {} songs removed".format(
            len(df) - len(remove_indices)))
        df = result
    df = df.drop(['dist'], axis=1)

    print("Class Distribution")
    print(df.y.value_counts())
    print("Labelled shape: {}".format(df.shape))
    print()
    return df


def main(*args):

    save = False
    data_size = -1
    spotify_thresh = 0
    deezer_thresh = 0
    class_per_quad = 1
    dist_mode = 'furthest_from_mean'

    for arg in args:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if k == 'save':
            save = str_to_bool(v)
        elif k == 'data_size':
            data_size = int(v)
        elif k == 'spotify_thresh':
            spotify_thresh = float(v)
        elif k == 'deezer_thresh':
            deezer_thresh = float(v)
        elif k == 'class_per_quad':
            class_per_quad = int(v)
        elif k == 'dist_mode':
            dist_mode = str(v)

    print()
    print("--- Clean config ---")
    print("save: {}".format(save))
    print("data_size: {}".format(data_size))
    print("spotify_thresh: {}".format(spotify_thresh))
    print("deezer_thresh: {}".format(deezer_thresh))
    print("class_per_quad: {}".format(class_per_quad))
    print("dist_mode: {}".format(dist_mode))
    print("--------------------")
    print()

    # SPOTIFY DATASET
    cleaned_df = clean(const.GEN_SPOTIFY)
    df = gen_labels(
        cleaned_df,
        cross_over_val=0.5,
        thresh=spotify_thresh,
        data_size=data_size,
        class_per_quadrant=class_per_quad,
        dist_mode=dist_mode,
    )
    check_dup(df, "Error: Duplicates!!!!!!!!!!!!!!!!!")

    if save:
        # save as csv (don't include indices in .csv)
        if data_size > 0:
            print("Saving result to {}...\n".format(const.CLEAN_SPOTIFY))
            df.to_csv(const.CLEAN_SPOTIFY, index=False)
        else:
            print("Saving result to {}...\n".format(const.CLEAN_UNEVEN_SPOTIFY))
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
        data_size=data_size,
        class_per_quadrant=class_per_quad,
        dist_mode=dist_mode,
    )
    check_dup(df, "Error: Duplicates!!!!!!!!!!!!!!!!!")
    if save:
        # save as csv (don't include indices in .csv)
        if data_size > 0:
            print("Saving result to {}...".format(const.CLEAN_DEEZER))
            df.to_csv(const.CLEAN_DEEZER, index=False)
        else:
            print("Saving result to {}...".format(const.CLEAN_UNEVEN_DEEZER))
            df.to_csv(const.CLEAN_UNEVEN_DEEZER, index=False)


if __name__ == '__main__':
    main(*sys.argv[1:])
