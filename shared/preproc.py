import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorize(csv_path, ngram_range, min_df, token_pattern=r'\b\w+\b'):
    '''  
    Uses TF-IDF vectorizer to convert lyric data to numeric features.
    Returns resulting data frame.
    '''
    df = pd.read_csv(csv_path)
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=token_pattern,
    )
    X = vectorizer.fit_transform(df.lyrics.values)
    X = pd.DataFrame(X.toarray())
    cols = [i for i in range(X.shape[0])]
    cols.append('y')
    return pd.concat([X, df.y], axis=1), vectorizer
