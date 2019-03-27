import pandas as pd
import numpy as np
import const
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import paired_euclidean_distances
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from utils import get_word_counts
from utils import get_unique_counts
from utils import get_emotion_counts
from utils import get_top_idf_ngrams
from utils import get_top_total_ngrams
from utils import get_class_based_data
from utils import get_stop_words
from utils import get_within_data_cos_similarity
from utils import get_between_class_cos_similarity
from utils import get_shared_ngrams

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
    scatter = plt.scatter(valence, arousal, alpha=0.7)
    plt.title(title)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")


print_ = True
show = False

gen_spotify_df = pd.read_csv(const.GEN_SPOTIFY)
clean_spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
if print_:
    print("Spotify missing per col: \n{}".format(clean_spotify_df.isna().sum()))

gen_deezer_df = pd.read_csv(const.GEN_DEEZER)
clean_deezer_df = pd.read_csv(const.CLEAN_DEEZER)
if print_:
    print("Deezer missing per col: \n{}".format(clean_deezer_df.isna().sum()))


# get info on datasets
clean_spotify_wc = get_word_counts(
    clean_spotify_df.lyrics.values, print_=print_)
clean_spotify_uc = get_unique_counts(
    clean_spotify_df.lyrics.values, print_=print_)
gen_spotify_wc = get_word_counts(gen_spotify_df.lyrics.values)
gen_spotify_uc = get_unique_counts(gen_spotify_df.lyrics.values)
spotify_class_distrib = get_emotion_counts(clean_spotify_df, print_=print_)
clean_deezer_wc = get_word_counts(clean_deezer_df.lyrics.values, print_=print_)
clean_deezer_uc = get_unique_counts(
    clean_deezer_df.lyrics.values, print_=print_)
gen_deezer_wc = get_word_counts(gen_deezer_df.lyrics.values)
gen_deezer_uc = get_unique_counts(gen_deezer_df.lyrics.values)
deezer_class_distrib = get_emotion_counts(clean_deezer_df, print_=print_)

# word count hist
gen_hist("Spotify vs Deezer: Dataset Word Count",
         clean_spotify_wc, "spotify", clean_deezer_wc, "deezer", a1=0.6, a2=0.6)


# unique word count hist
gen_hist("Spotify vs Deezer: Dataset Unique Words Count",
         clean_spotify_uc, "spotify", clean_deezer_uc, "deezer", a1=0.6, a2=0.6)

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


def get_y(df):
    y = [750] * 4
    for i in df.y.unique():
        y[int(i)-1] = len(df.loc[df.y == i])
    return y


# class distrib hist
plt.figure()
x = np.array([1, 2, 3, 4])

plt.title("Spotify vs Deezer: Class Distribution")
plt.bar(x - 0.125, get_y(clean_spotify_df),
        width=0.25, align='center', label='spotify')
plt.bar(x + 0.125, get_y(clean_deezer_df),
        width=0.25, align='center', label='deezer')
plt.xticks(x)
plt.legend()
if show:
    plt.show()
