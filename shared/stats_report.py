import pandas as pd
import numpy as np
import const
import matplotlib.pyplot as plt


def get_lyric_wordcount(csv_path):
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
    print("Lyric word count info - '{}'".format(csv_path))
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


# debug function
def print_song(df, i):
    print()
    print("song: '{}' by {}".format(df.song[i], df.artist[i]))
    print("lyrics: {}".format(df.lyrics[i]))
    print("y: {}".format(df.y[i]))


deezer_df = pd.read_csv(const.CLEAN_DEEZER)
spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)

spotify_wc, spotify_uc = get_lyric_wordcount(const.CLEAN_SPOTIFY)
spotify_class_distrib = get_emotion_info(const.CLEAN_SPOTIFY)


deezer_wc, deezer_uc = get_lyric_wordcount(const.CLEAN_DEEZER)
deezer_class_distrib = get_emotion_info(const.CLEAN_DEEZER)

# word count hist
spotify_word_count = plt.subplot(1, 1, 1)
spotify_word_count.hist(spotify_wc, alpha=0.7, bins=100)

deezer_word_count = plt.subplot(1, 1, 1)
deezer_word_count.hist(deezer_wc, alpha=0.7, bins=100)

# unique word count hist
plt.figure()
spotify_unique_count = plt.subplot(1, 1, 1)
spotify_unique_count.hist(spotify_uc, alpha=0.7, bins=100)

deezer_unique_count = plt.subplot(1, 1, 1)
deezer_unique_count.hist(deezer_uc, alpha=0.7, bins=100)

# class distrib scatter plot
plt.figure()
spotify_val_arous = plt.subplot(1, 1, 1)
spotify_val_arous.scatter(spotify_df.valence, spotify_df.arousal, alpha=0.7)

plt.figure()
deezer_val_arous = plt.subplot(1, 1, 1)
deezer_val_arous.scatter(deezer_df.valence, deezer_df.arousal, alpha=0.7)

# class distrib hist
plt.figure()
spotify_class_hist = plt.subplot(1, 1, 1)
spotify_class_hist.hist(spotify_df.y, alpha=0.7)

deezer_class_hist = plt.subplot(1, 1, 1)
deezer_class_hist.hist(deezer_df.y, alpha=0.7)

plt.show()
