import pandas as pd
import numpy as np
import lyricsgenius
import progressbar

ACCESS_TOKEN = "u330dU6EonyHdWwl1_3PxPgbCStdO_4lHFU6GGSQ-DPBHfTphtfkVdm4WfyLUcug"
genius = lyricsgenius.Genius(ACCESS_TOKEN)


SONG_IDX = 0
ARTIST_IDX = 1
VAL_IDX = 2
AROUS_IDX = 3
LYRICS_IDX = 4

columns = [
    "song",
    "artist",
    "valence",
    "arousal",
    "lyrics"
]


def fetch_lyrics(df, song_col, artist_col, valence_col, arousal_col, csv_name, n_songs=-1, start_from=0):

    print("Fetching lyrics for {} dataset".format(csv_name))

    # create result data frame
    result = np.empty(shape=(df.shape[0], 5), dtype=object)
    result_df = pd.DataFrame(result, columns=columns)

    # load previous results if not starting from 1st
    if start_from != 0:
        result_df = pd.read_csv(csv_name)

    search_range = n_songs if n_songs != -1 else df.shape[0]
    n_songs = search_range - start_from
    missing = 0
    space = ' '
    line_break = '\n'

    # fetch song lyrics from genius
    for i in progressbar.progressbar(range(start_from, search_range)):
        print()
        song_name = df[song_col][i]
        artist = df[artist_col][i]
        result_df.iloc[i, SONG_IDX] = song_name
        result_df.iloc[i, ARTIST_IDX] = artist
        result_df.iloc[i, VAL_IDX] = df[valence_col][i]
        result_df.iloc[i, AROUS_IDX] = df[arousal_col][i]
        song = None

        # try/catch for timeout exception
        try:
            song = genius.search_song(song_name, artist)
        except:
            print("exception caught!")

        if song != None:
            lyrics = song.lyrics.replace(line_break, space)
            result_df.iloc[i, LYRICS_IDX] = lyrics
        else:
            missing += 1

        # save progress every 100 songs
        if i % 100 == 0:
            result_df.to_csv(csv_name, index=False)

    # save to file
    result_df.to_csv(csv_name, index=False)
    print("Finished fetching lyrics ({} out of {})".format(
        n_songs - missing, search_range))
    print()
    return result_df


def get_start_from(csv_name, save_incr=100):
    # returns the first index of row that is all NaN
    prev = pd.read_csv(csv_name)
    na = prev.isna().all(axis=1)
    last = na[na == True].index[0]
    return last - (last % save_incr)


# SPOTIFY DATASET
# spotify_data = pd.read_csv("spotify/song_data.csv")
# spotify_info = pd.read_csv("spotify/song_info.csv")
# spotify_info = spotify_info.drop("song_name", axis=1)
# spotify_df = pd.concat([spotify_data, spotify_info], axis=1)
# spotify_result = fetch_lyrics(spotify_df, "song_name", "artist_name",
#                               "audio_valence", "energy", "spotify_data.csv", -1,
#                               get_start_from("spotify_data.csv"))


# # DEEZER DATASET
deezer_df = pd.read_csv("deezer/test.csv")
deezer_df = deezer_df.append(pd.read_csv(
    "deezer/train.csv"), ignore_index=True)
deezer_df = deezer_df.append(pd.read_csv(
    "deezer/validation.csv"), ignore_index=True)

deezer_result = fetch_lyrics(deezer_df, "track_name", "artist_name",
                             "valence", "arousal", "deezer_data.csv", -1)
