import os
from pathlib import Path

# shared constant defintions

# base dir abs path
BASE_DIR = str(Path(os.path.abspath(__file__)).parent.parent)

DEEZER = "deezer"
SPOTIFY = "spotify"

DATA_DIR = BASE_DIR+"/data"
# original datasets
SPOTIFY_DIR = DATA_DIR+"/spotify"
SPOTIFY_DATA = SPOTIFY_DIR+"/song_data.csv"
SPOTIFY_INFO = SPOTIFY_DIR+"/song_info.csv"

DEEZER_DIR = DATA_DIR+"/deezer"
DEEZER_TEST = DEEZER_DIR+"/test.csv"
DEEZER_TRAIN = DEEZER_DIR+"/train.csv"
DEEZER_VALID = DEEZER_DIR+"/validation.csv"

# generated datasets
GEN_DIR = DATA_DIR+"/gen"
GEN_SPOTIFY = GEN_DIR+"/gen_spotify_data.csv"
GEN_SPOTIFY_LOG = GEN_DIR+"/gen_spotify_error_log.txt"
GEN_DEEZER = GEN_DIR+"/gen_deezer_data.csv"
GEN_DEEZER_LOG = GEN_DIR+"/gen_deezer_error_log.txt"

# cleaned datasets
CLEAN_DIR = DATA_DIR+"/clean"
CLEAN_SPOTIFY = CLEAN_DIR+"/clean_spotify_data.csv"
CLEAN_DEEZER = CLEAN_DIR + "/clean_deezer_data.csv"
CLEAN_UNEVEN_SPOTIFY = CLEAN_DIR+"/clean_uneven_spotify.csv"
CLEAN_UNEVEN_DEEZER = CLEAN_DIR+"/clean_uneven_deezer.csv"

# dataset column names
COL_SONG = "song"
COL_ARTIST = "artist"
COL_VALENCE = "valence"
COL_AROUSAL = "arousal"
COL_LYRICS = "lyrics"
COL_FOUND_SONG = "found_song"
COL_FOUND_ARTIST = "found_artist"

# implementation models
MODEL_DIR = BASE_DIR+"/models"
ADABOOST_MODEL_DIR = MODEL_DIR+"/adaboost"


GSCV_MODE = "gs"
EVAL_MODE = "eval"
COS_SIM_MODE = "cos_sim"
WORDCLOUD_MODE = "wordcloud"

# some shared default config values
EVEN_DISTRIB_DEFAULT = False
PLOT_DEFAULT = False
PRINT_DEFAULT = True
RANDOM_STATE_DEFAULT = 0
SAVE_DEFAULT = False
FONT_SIZE_DEFAULT = 22
