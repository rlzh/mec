import pandas as pd
import numpy as np
import const


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


spotify_wc, spotify_uc = get_lyric_wordcount(const.CLEAN_SPOTIFY)
spotify_class_distrib = get_emotion_info(const.CLEAN_SPOTIFY)


deezer_wc, deezer_uc = get_lyric_wordcount(const.CLEAN_DEEZER)
deezer_class_distrib = get_emotion_info(const.CLEAN_DEEZER)
