import pandas as pd
import numpy as np
import const
import nltk
import sys
import utils
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import paired_euclidean_distances
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from utils import get_word_counts
from utils import get_unique_counts
from utils import get_emotion_counts
from utils import get_class_based_data
from utils import get_stop_words

# load stop words
stop_words = get_stop_words()


def gen_word_cloud(top_n):
    plt.figure()
    wordcloud = WordCloud().generate_from_frequencies(frequencies=top_n)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")


def gen_hist(title, hist1, label1, hist2=np.array([None]), label2="", a1=1, a2=0.25, bins=100):
    fig = plt.figure()
    hist = plt.hist(hist1, alpha=a1, bins=100, label=label1)
    plt.title(title)
    if hist2.all() != None:
        plt.hist(hist2, alpha=a2, bins=100, label=label2)
        plt.legend()


def gen_val_arousal_scatter(title, valence, arousal):
    fig = plt.figure()
    scatter = plt.scatter(valence, arousal, alpha=0.6)
    plt.title(title)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")


def get_y(df):
    y = [0] * len(df.y.unique())
    for i in df.y.unique():
        y[int(i)-1] = len(df.loc[df.y == i])
    return y


def main(*args):

    plot = const.PLOT_DEFAULT
    print_ = const.PRINT_DEFAULT
    even_distrib = const.EVEN_DISTRIB_DEFAULT
    plt.rcParams.update({'font.size': const.FONT_SIZE_DEFAULT})
    # print command line arguments
    for arg in args:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if k == 'plot':
            plot = utils.str_to_bool(v)
        elif k == 'print':
            print_ = utils.str_to_bool(v)
        elif k == 'font_size':
            plt.rcParams.update({'font.size': int(v)})
        elif k == 'even_distrib':
            even_distrib = utils.str_to_bool(v)

    if print_:
        print()
        print("--- Stats config ---")
        print("Even distribution dataset: {}".format(even_distrib))
        print("Plot: {}".format(plot))
        print("--------------------")
        print()

    # load data
    gen_spotify_df = pd.read_csv(const.GEN_SPOTIFY)
    clean_spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
    if even_distrib == False:
        clean_spotify_df = pd.read_csv(const.CLEAN_UNEVEN_SPOTIFY)
    if print_:
        print("Spotify missing per col: \n{}".format(
            clean_spotify_df.isna().sum()))
        print()

    gen_deezer_df = pd.read_csv(const.GEN_DEEZER)
    clean_deezer_df = pd.read_csv(const.CLEAN_DEEZER)
    if even_distrib == False:
        clean_deezer_df = pd.read_csv(const.CLEAN_UNEVEN_DEEZER)
    if print_:
        print("Deezer missing per col: \n{}".format(
            clean_deezer_df.isna().sum()))
        print()

    # get info on datasets
    clean_spotify_wc = get_word_counts(
        clean_spotify_df.lyrics.values, print_=print_)
    clean_spotify_uc = get_unique_counts(
        clean_spotify_df.lyrics.values, print_=print_)
    spotify_class_distrib = get_emotion_counts(clean_spotify_df, print_=print_)
    clean_deezer_wc = get_word_counts(
        clean_deezer_df.lyrics.values, print_=print_)
    clean_deezer_uc = get_unique_counts(
        clean_deezer_df.lyrics.values, print_=print_)
    deezer_class_distrib = get_emotion_counts(clean_deezer_df, print_=print_)

    # word count hist
    gen_hist("Spotify vs Deezer: Dataset Word Count",
             clean_spotify_wc, const.SPOTIFY, clean_deezer_wc, const.DEEZER, a1=0.4, a2=0.4)

    # unique word count hist
    gen_hist("Spotify vs Deezer: Dataset Unique Words Count",
             clean_spotify_uc, const.SPOTIFY, clean_deezer_uc, const.DEEZER, a1=0.4, a2=0.4)

    # class distrib scatter plot
    gen_val_arousal_scatter(
        "Spotify: Class Distribution",
        clean_spotify_df.valence.values,
        clean_spotify_df.arousal.values
    )
    gen_val_arousal_scatter(
        "Deezer: Class Distribution",
        clean_deezer_df.valence.values,
        clean_deezer_df.arousal.values
    )

    # class distrib hist
    plt.figure()
    x = np.array([1, 2, 3, 4])

    plt.title("Spotify vs Deezer: Class Distribution")
    plt.bar(x - 0.125, get_y(clean_spotify_df),
            width=0.25, align='center', label=const.SPOTIFY)
    plt.bar(x + 0.125, get_y(clean_deezer_df),
            width=0.25, align='center', label=const.DEEZER)
    plt.xticks(x, ["Happy", "Angry", "Sad", "Relaxed"])
    plt.legend()
    if plot:
        plt.show()


if __name__ == '__main__':
    main(*sys.argv[1:])
