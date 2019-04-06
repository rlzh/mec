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


def plot_hist(title, hist1, label1, hist2=np.array([None]), label2="", a1=1, a2=0.25, bins=100, xlabel="", ylabel=""):
    fig = plt.figure()
    hist = plt.hist(hist1, alpha=a1, bins=100, label=label1)
    plt.title(title)
    plt.xlabel(xlabel)
    if hist2.all() != None:
        plt.hist(hist2, alpha=a2, bins=100, label=label2)
        plt.legend()
        plt.ylabel(ylabel)


def plot_val_arousal_scatter(title, valence, arousal, unclean_valence=np.array([None]), unclean_arousal=np.array([None])):
    fig = plt.figure()
    label = ""
    if unclean_arousal.all() != None and unclean_valence.all() != None:
        scatter = plt.scatter(unclean_valence, unclean_arousal,
                              alpha=0.1, label="Uncleaned")
        label = "Cleaned"
    scatter = plt.scatter(valence, arousal, alpha=1.0, label=label)

    plt.title(title)
    plt.legend()
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
        print("Spotify unclean shape: {}".format(gen_spotify_df))
        print("Spotify shape: {}".format(clean_spotify_df))
        print()

    gen_deezer_df = pd.read_csv(const.GEN_DEEZER)
    clean_deezer_df = pd.read_csv(const.CLEAN_DEEZER)
    if even_distrib == False:
        clean_deezer_df = pd.read_csv(const.CLEAN_UNEVEN_DEEZER)
    if print_:
        print("Deezer missing per col: \n{}".format(
            clean_deezer_df.isna().sum()))
        print("Spotify unclean shape: {}".format(gen_deezer_df))
        print("Deezer shape: {}".format(clean_deezer_df))
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
    plot_hist("Dataset Word Count",
              clean_spotify_wc, const.SPOTIFY, clean_deezer_wc, const.DEEZER, a1=0.4, a2=0.4,
              xlabel="# of Songs", ylabel="Word Count")

    # unique word count hist
    plot_hist("Dataset Unique Words Count",
              clean_spotify_uc, const.SPOTIFY, clean_deezer_uc, const.DEEZER, a1=0.4, a2=0.4,
              ylabel="Unique Word Count", xlabel="# of Songs")

    # class distrib scatter plot
    plot_val_arousal_scatter(
        "Spotify: Valence-Arousal Distribution",
        clean_spotify_df.valence.values,
        clean_spotify_df.arousal.values,
        gen_spotify_df.valence.values,
        gen_spotify_df.arousal.values,
    )
    plot_val_arousal_scatter(
        "Deezer: Valence-Arousal Distribution",
        clean_deezer_df.valence.values,
        clean_deezer_df.arousal.values,
        gen_deezer_df.valence.values,
        gen_deezer_df.arousal.values,
    )

    datasets = [
        (const.SPOTIFY, clean_spotify_df),
        (const.DEEZER, clean_deezer_df),
    ]
    for name, dataset in datasets:
        for i in dataset.y.unique():
            class_df = utils.get_class_based_data(
                dataset,
                i,
                random_state=const.RANDOM_STATE_DEFAULT,
                include_other_classes=True,
                even_distrib=False,
                limit_size=False,
                print_=True
            )
            print("{} Class {} data shape: {}".format(name, i, class_df.shape))
            print("{} Class {} data mean valence-arousal: {}".format(
                name, i, (class_df.valence.mean(), class_df.arousal.mean())))
            plot_val_arousal_scatter(
                "{}: Class {} Data Valence-Arousal Distribution".format(
                    name, i),
                class_df.valence.values,
                class_df.arousal.values
            )

    # class distrib hist
    plt.figure()
    x = np.array([i+1 for i in range(len(clean_spotify_df.y.unique()))])

    plt.title("Dataset Class Distribution")
    plt.bar(x - 0.125, get_y(clean_spotify_df),
            width=0.25, align='center', label=const.SPOTIFY)
    plt.bar(x + 0.125, get_y(clean_deezer_df),
            width=0.25, align='center', label=const.DEEZER)
    plt.xticks(x, labels=["Happy", "Angry", "Sad", "Relaxed"])
    plt.legend()
    if plot:
        plt.draw()
        plt.show()


if __name__ == '__main__':
    main(*sys.argv[1:])
