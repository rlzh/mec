import pandas as pd
import numpy as np
import const
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from utils import get_word_counts
from utils import get_unique_counts
from utils import get_emotion_counts
from utils import get_top_tfidf_ngrams
from utils import get_class_based_data
from utils import get_stop_words

# load stop words
stop_words = get_stop_words()


def gen_word_clouds(csv_path, vectorizer, n=10):
    df = pd.read_csv(csv_path)
    for i in df.y.unique():
        class_df = get_class_based_data(
            df,
            i,
            random_state=np.random.seed(0),
            include_other_classes=False,
            limit_size=False,
        )
        vectorized = vectorizer.fit_transform(
            class_df[const.COL_LYRICS].values)
        top_n = get_top_tfidf_ngrams(vectorized, vectorizer, n=n, scaled=True)
        print("Top {} for Class {}\n  {}".format(n, i, top_n))
        plt.figure()
        plt.title('Top {} n-grams for Class {} ({})'.format(n, i, type(vectorizer)))
        wordcloud = WordCloud().generate_from_frequencies(frequencies=top_n)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")


# def print_song(df, i):
#     # debug function
#     print()
#     print("song: '{}' by {}".format(df.song[i], df.artist[i]))
#     print("lyrics: {}".format(df.lyrics[i]))
#     print("y: {}".format(df.y[i]))


deezer_df = pd.read_csv(const.CLEAN_DEEZER)
print("Deezer missing per col: \n{}".format(deezer_df.isna().sum()))

spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
print("Spotify missing per col: \n{}".format(spotify_df.isna().sum()))

# get info on datasets
spotify_wc = get_word_counts(spotify_df[const.COL_LYRICS].values)
spotify_uc = get_unique_counts(spotify_df[const.COL_LYRICS].values)
spotify_class_distrib = get_emotion_counts(spotify_df)
deezer_wc = get_word_counts(deezer_df[const.COL_LYRICS].values)
deezer_uc = get_unique_counts(deezer_df[const.COL_LYRICS].values)
deezer_class_distrib = get_emotion_counts(deezer_df)

# word count hist
plt.figure()
plt.hist(spotify_wc, alpha=0.7, bins=100)
plt.hist(deezer_wc, alpha=0.7, bins=100)

# unique word count hist
plt.figure()
plt.hist(spotify_uc, alpha=0.7, bins=100)
plt.hist(deezer_uc, alpha=0.7, bins=100)

# class distrib scatter plot
plt.figure()
plt.scatter(spotify_df.valence, spotify_df.arousal, alpha=0.7)
plt.figure()
plt.scatter(deezer_df.valence, deezer_df.arousal, alpha=0.7)

# class distrib hist
plt.figure()
plt.hist(spotify_df.y, alpha=0.7)
plt.hist(deezer_df.y, alpha=0.7)


# word clouds
tfidf = TfidfVectorizer(
    stop_words=stop_words,
    min_df=10,
    ngram_range=(1, 1),
)
count = CountVectorizer(
    stop_words=stop_words,
    min_df=10,
    ngram_range=(1, 1),
)

gen_word_clouds(
    const.CLEAN_SPOTIFY,
    vectorizer=tfidf,
    n=50
)

plt.show()
