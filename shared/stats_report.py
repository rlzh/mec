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


def gen_hist(title, clean, gen=np.array([None]), bins=100):
    fig = plt.figure()
    hist = plt.hist(clean, alpha=1, bins=100, label="cleaned")
    plt.title(title)
    if gen.all() != None:
        plt.hist(gen, alpha=0.25, bins=100, label="uncleaned")
        plt.legend()


def gen_val_arousal_scatter(title, valence, arousal):
    fig = plt.figure()
    scatter = plt.scatter(valence, arousal, alpha=0.7)
    plt.title(title)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")


print_ = True
show = True

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
gen_hist("Spotify: Dataset Word Count (clean vs gen)",
         clean_spotify_wc, gen_spotify_wc)
gen_hist("Spotify Dataset Word Count", clean_spotify_wc)
gen_hist("Deezer: Dataset Word Count (clean vs gen)",
         clean_deezer_wc, gen_deezer_wc)
gen_hist("Deezer: Dataset Word Count", clean_deezer_wc)

# unique word count hist
gen_hist("Spotify: Dataset Unique Words Count (clean vs gen)",
         clean_spotify_uc, gen_spotify_uc)
gen_hist("Spotify: Dataset Unique Words Count", clean_spotify_uc)
gen_hist("Deezer: Dataset Unique Words Count (clean vs gen)",
         clean_deezer_uc, gen_deezer_uc)
gen_hist("Deezer: Dataset Unique Words Count", clean_deezer_uc)

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
plt.title("Spotify: Class Distribution")
plt.hist(clean_spotify_df.y, alpha=0.7)
plt.figure()
plt.title("Deezer: Class Distribution")
plt.hist(clean_deezer_df.y, alpha=0.7)

if show:
    plt.show()
