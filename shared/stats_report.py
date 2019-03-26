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
from utils import get_top_idf_ngrams
from utils import get_top_singular_ngrams
from utils import get_top_total_ngrams
from utils import get_class_based_data
from utils import get_stop_words

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


def gen_word_clouds(name, df, vectorizer, top_n_generator, n=10):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{}: Top {} n-grams ({})'.format(name, n, type(vectorizer)))
    top_n_list = []
    for i in df.y.unique():
        class_df = get_class_based_data(
            df,
            i,
            random_state=np.random.seed(0),
            include_other_classes=False,
            limit_size=False,
        )
        vectorized = vectorizer.fit_transform(class_df.lyrics.values)
        top_n = top_n_generator(
            vectorized,
            vectorizer,
            n=n
        )
        print("Top {} for Class {}\n  {}".format(n, i, top_n))
        wordcloud = WordCloud().generate_from_frequencies(frequencies=top_n)
        ax = axs[0, 0]
        ax.set_title('Angry')
        if i == 1:
            ax = axs[0, 1]
            ax.set_title('Happy')
        elif i == 3:
            ax = axs[1, 0]
            ax.set_title('Sad')
        elif i == 4:
            ax = axs[1, 1]
            ax.set_title('Relaxed')
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        keys = []
        for key in top_n:
            keys.append(key)
        top_n_list.append(keys)
    fig.tight_layout()
    return np.array(top_n_list)


def get_shared_ngrams(data):
    shared_grams = set()
    unique_grams = set()
    for row in data:
        for word in row:
            if word in shared_grams:
                continue
            unique_grams.add(word)
            is_shared = True
            for other_row in data:
                if word not in other_row:
                    is_shared = False
                    break
            if is_shared:
                shared_grams.add(word)
    return shared_grams, unique_grams

# def print_song(df, i):
#     # debug function
#     print()
#     print("song: '{}' by {}".format(df.song[i], df.artist[i]))
#     print("lyrics: {}".format(df.lyrics[i]))
#     print("y: {}".format(df.y[i]))


gen_spotify_df = pd.read_csv(const.GEN_SPOTIFY)
clean_spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
print("Spotify missing per col: \n{}".format(clean_spotify_df.isna().sum()))

gen_deezer_df = pd.read_csv(const.GEN_DEEZER)
clean_deezer_df = pd.read_csv(const.CLEAN_DEEZER)
print("Deezer missing per col: \n{}".format(clean_deezer_df.isna().sum()))

# get info on datasets
clean_spotify_wc = get_word_counts(clean_spotify_df.lyrics.values, print_=True)
clean_spotify_uc = get_unique_counts(
    clean_spotify_df.lyrics.values, print_=True)
gen_spotify_wc = get_word_counts(gen_spotify_df.lyrics.values)
gen_spotify_uc = get_unique_counts(gen_spotify_df.lyrics.values)
spotify_class_distrib = get_emotion_counts(clean_spotify_df, print_=True)
clean_deezer_wc = get_word_counts(clean_deezer_df.lyrics.values, print_=True)
clean_deezer_uc = get_unique_counts(clean_deezer_df.lyrics.values, print_=True)
gen_deezer_wc = get_word_counts(gen_deezer_df.lyrics.values)
gen_deezer_uc = get_unique_counts(gen_deezer_df.lyrics.values)
deezer_class_distrib = get_emotion_counts(clean_deezer_df, print_=True)

# # word count hist
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
plt.hist(clean_spotify_df.y, alpha=0.7)
plt.hist(clean_deezer_df.y, alpha=0.7)


# word clouds
n = 5000
tfidf = TfidfVectorizer(
    stop_words=stop_words,
    # min_df=1,
    ngram_range=(1, 1),
)
top_n = gen_word_clouds(
    'Spotify',
    clean_spotify_df,
    vectorizer=tfidf,
    top_n_generator=get_top_idf_ngrams,
    n=n
)
spotify_tfidf_shared, spotify_tfidf_unique = get_shared_ngrams(top_n)

top_n = gen_word_clouds(
    'Deezer',
    clean_deezer_df,
    vectorizer=tfidf,
    top_n_generator=get_top_idf_ngrams,
    n=n
)
deezer_tfidf_shared, deezer_tfidf_unique = get_shared_ngrams(top_n)

count = CountVectorizer(
    stop_words=stop_words,
    max_df=0.25,
    ngram_range=(1, 1),
)
top_n = gen_word_clouds(
    'Spotify',
    clean_spotify_df,
    vectorizer=count,
    top_n_generator=get_top_total_ngrams,
    n=n
)
spotify_count_shared, spotify_count_unique = get_shared_ngrams(top_n)

top_n = gen_word_clouds(
    'Deezer',
    clean_deezer_df,
    vectorizer=count,
    top_n_generator=get_top_total_ngrams,
    n=n
)
deezer_count_shared, deezer_count_unique = get_shared_ngrams(top_n)

# test1 = ["this", "is", "a", "test"]
# test2 = ["this", "is", "a", "another"]
# test = np.array([test1, test1, test2, ])
# shared, unique = get_shared_ngrams(test)
# print(len(shared)/len(unique))

print()
print("Spotify: tfidf shared={}".format(
    len(spotify_tfidf_shared)/len(spotify_tfidf_unique)))
print("Deezer: tfidf shared={}".format(
    len(deezer_tfidf_shared)/len(deezer_tfidf_unique)))
print("Spotify: count shared={}".format(
    len(spotify_count_shared)/len(spotify_count_unique)))
print("Deezer: count shared={}".format(
    len(deezer_count_shared)/len(deezer_count_unique)))
print()

# vectorized = count.fit_transform(clean_spotify_df.lyrics.values)
# gen_word_cloud(get_top_total_ngrams(
#     vectorized,
#     count.get_feature_names(),
#     n=n
# ))

# vectorized = count.fit_transform(clean_deezer_df.lyrics.values)
# gen_word_cloud(get_top_total_ngrams(
#     vectorized,
#     count.get_feature_names(),
#     n=n
# ))

plt.show()
