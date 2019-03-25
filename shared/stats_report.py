import pandas as pd
import numpy as np
import const
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from preproc import get_class_based_data
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler

# load stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
f = open("stopwords.txt", 'r')
for l in f:
    if len(l.strip()) > 0:
        stop_words.add(l.strip())


def get_lyrics_info(csv_path):
    df = pd.read_csv(csv_path)
    word_count = np.empty(shape=(df.shape[0],))
    unique_count = np.empty(shape=(df.shape[0],))
    for i in range(df.shape[0]):
        words = df.lyrics[i].split(' ')
        word_count[i] = len(words)
        s = set()
        for w in words:
            if w not in s:
                s.add(w)
                unique_count[i] += 1
    print("Lyrics info - '{}'".format(csv_path))
    print("min word count: {} (id: {})".format(
        np.min(word_count), np.argmin(word_count)))
    print("max word count: {} (id: {})".format(
        np.max(word_count), np.argmax(word_count)))
    print("ave word count: {}".format(np.mean(word_count)))
    print()
    print("unique count")
    print("min unique count: {} (id: {})".format(
        np.min(unique_count), np.argmin(unique_count)))
    print("max unique count: {} (id: {})".format(
        np.max(unique_count), np.argmax(unique_count)))
    print("ave unique count: {}".format(np.mean(unique_count)))
    print()

    # print_song(df, np.argmax(word_count))

    return word_count, unique_count


def get_emotion_info(csv_path):
    df = pd.read_csv(csv_path)
    class_distrib = df['y'].value_counts()
    print("Emotion info - '{}'".format(csv_path))
    print("valence range: [{},{}]".format(df.valence.min(), df.valence.max()))
    print("arousal range: [{},{}]".format(df.arousal.min(), df.arousal.max()))
    print(class_distrib)
    print()
    return class_distrib


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
        top_n = get_top_n_grams(class_df.lyrics.values,
                                vectorizer=vectorizer, n=n)
        print("Top {} for Class ({})\n  {}".format(n, i, top_n))
        plt.figure()
        plt.title('Top {} n-grams for Class ({})'.format(n, i))
        wordcloud = WordCloud().generate_from_frequencies(frequencies=top_n)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")


def get_top_n_grams(lyrics, vectorizer, n=10):
    vectorized = vectorizer.fit_transform(lyrics)
    feature_array = np.array(vectorizer.get_feature_names())
    vectorized_sorted = np.sort(vectorized.toarray().flatten())[::-1]
    vectorized_sorting = np.argsort(vectorized.toarray()).flatten()[::-1]
    top_n = feature_array[vectorized_sorting][:n]
    top_n_scores = MinMaxScaler().fit_transform(
        vectorized_sorted[:n].reshape(-1, 1))
    top_n_scores = np.add(top_n_scores, 1.0).reshape(-1,)
    top_n = dict(zip(top_n, top_n_scores))

    return top_n


# debug
def print_song(df, i):
    print()
    print("song: '{}' by {}".format(df.song[i], df.artist[i]))
    print("lyrics: {}".format(df.lyrics[i]))
    print("y: {}".format(df.y[i]))


deezer_df = pd.read_csv(const.CLEAN_DEEZER)
print("Deezer missing per col: \n{}".format(deezer_df.isna().sum()))

spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
print("Spotify missing per col: \n{}".format(spotify_df.isna().sum()))

# get info on datasets
spotify_wc, spotify_uc = get_lyrics_info(const.CLEAN_SPOTIFY)
spotify_class_distrib = get_emotion_info(const.CLEAN_SPOTIFY)
deezer_wc, deezer_uc = get_lyrics_info(const.CLEAN_DEEZER)
deezer_class_distrib = get_emotion_info(const.CLEAN_DEEZER)

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
    min_df=5,
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
