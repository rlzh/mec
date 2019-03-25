import pandas as pd
import numpy as np
import lyricsgenius
import progressbar
import re
import string
import const

ACCESS_TOKEN = "u330dU6EonyHdWwl1_3PxPgbCStdO_4lHFU6GGSQ-DPBHfTphtfkVdm4WfyLUcug"
genius = lyricsgenius.Genius(ACCESS_TOKEN)
genius.excluded_terms = ["(Remix)", "(Live)"]
genius.skip_non_songs = True

SAVE_INCR = 100

SONG_IDX = 0
ARTIST_IDX = 1
VAL_IDX = 2
AROUS_IDX = 3
LYRICS_IDX = 4
FOUND_SONG_IDX = 5
FOUND_ARTIST_IDX = 6


columns = [
    const.COL_SONG,
    const.COL_ARTIST,
    const.COL_VALENCE,
    const.COL_AROUSAL,
    const.COL_LYRICS,
    const.COL_FOUND_SONG,
    const.COL_FOUND_ARTIST,
]


def fetch_lyrics(dataset_name, df, song_col, artist_col, valence_col, arousal_col, csv_name, err_log_name, n_songs=-1, start_from=0):

    print("Fetching lyrics for {} dataset".format(csv_name))

    # create log file for errors
    permission = 'w' if start_from == 0 else 'a'
    f = open(err_log_name, permission)
    # create result data frame
    result = np.empty(shape=(df.shape[0], 7), dtype=object)
    result_df = pd.DataFrame(result, columns=columns)

    # load previous results if not starting from 1st
    if start_from != 0:
        result_df = pd.read_csv(csv_name)

    search_range = n_songs if n_songs != -1 else df.shape[0]
    n_songs = search_range - start_from
    missing = 0
    space = ' '
    line_break = '\n'
    # regex pattern to remove unnecessary tags from song title (e.g. - Remastered Version)
    pattern = '(-.*(Remaster(ed)?|Mix|Version|Acoustic).*)|(\(?((F|f)eat|(A|a)coustic).*)'
    punctuations_space = string.punctuation + space

    # fetch song lyrics from genius
    for i in progressbar.progressbar(range(start_from, search_range)):
        print()
        song_name = re.sub(pattern, '', df[song_col][i]).strip()
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
            pass

        if song != None:
            lyrics = song.lyrics.replace(line_break, space)
            found_artist = song.artist.lower().strip()
            found_artist = found_artist.replace(u'\xa0', u' ')
            found_song_name = song.title.lower().strip()
            found_song_name = found_song_name.replace(punctuations_space, '')
            found_song_name = found_song_name.replace(u'\xa0', u' ')
            artist_lower = artist.lower().strip()
            song_name_lower = song_name.lower().strip()
            song_name_lower = song_name_lower.replace(punctuations_space, '')
            # validate search result based on artist and song title
            if (found_artist in artist_lower or artist_lower in found_artist) and (found_song_name in song_name_lower or song_name_lower in found_song_name):
                result_df.iloc[i, LYRICS_IDX] = lyrics
                result_df.iloc[i, FOUND_SONG_IDX] = song.title
                result_df.iloc[i, FOUND_ARTIST_IDX] = str(song.artist)
            else:
                err = "Invalid result for '{}' by {}. Found '{}' by {} instead\n".format(
                    song_name, artist, song.title, song.artist)
                print(err)
                f.write(err)
                missing += 1
        else:
            err = "No result for '{}' by {}\n".format(song_name, artist)
            f.write(err)
            missing += 1

        # save progress every SAVE_INCR songs
        if i % SAVE_INCR == 0:
            result_df.to_csv(csv_name, index=False)
            f.flush()

    # save to file
    result_df.to_csv(csv_name, index=False)
    print("Finished fetching lyrics ({} out of {})".format(
        n_songs - missing, search_range))
    print()

    # close error log file
    f.close()
    return result_df


def get_start_from(csv_name):
    # returns the first index of row that is all NaN
    prev = pd.read_csv(csv_name)
    na = prev.isna().all(axis=1)
    last = na[na == True].index[0]
    return last - (last % SAVE_INCR)


# SPOTIFY DATASET
spotify_data = pd.read_csv(const.SPOTIFY_DATA)
spotify_info = pd.read_csv(const.SPTOFY_INFO)
spotify_info = spotify_info.drop("song_name", axis=1)
spotify_df = pd.concat([spotify_data, spotify_info], axis=1)
spotify_result = fetch_lyrics(
    'spotify',
    spotify_df,
    "song_name",
    "artist_name",
    "audio_valence",
    "energy",
    const.GEN_SPOTIFY,
    const.GEN_SPOTIFY_LOG,
    -1,
    # get_start_from(const.GEN_SPOTIFY)
)


# DEEZER DATASET
deezer_df = pd.read_csv(const.DEEZER_TEST)
deezer_df = deezer_df.append(pd.read_csv(
    const.DEEZER_TRAIN), ignore_index=True)
deezer_df = deezer_df.append(pd.read_csv(
    const.DEEZER_VALID), ignore_index=True)
deezer_result = fetch_lyrics(
    'deezer',
    deezer_df,
    "track_name",
    "artist_name",
    "valence",
    "arousal",
    GEN_DEEZER,
    GEN_DEEZER_LOG,
    -1,
    # get_start_from(const.GEN_DEEZER)
)
