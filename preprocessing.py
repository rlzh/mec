import pandas as pd

spotify_df = pd.read_csv("spotify_data.csv")
print(spotify_df.shape)
print(spotify_df.columns)
print(spotify_df["lyrics"].isna().sum())
spotify_df = spotify_df.dropna()

deezer_df = pd.read_csv("deezer_data.csv")
print(deezer_df.shape)
print(deezer_df.columns)
print(deezer_df["lyrics"].isna().sum())
spotify_df = deezer_df.dropna()
