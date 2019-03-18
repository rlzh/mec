import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorize(df, ngram_range, min_df, token_pattern=r'\b\w+\b'):
    '''  
    Uses TF-IDF vectorizer to convert lyric data to numeric features.
    Returns resulting data frame.
    '''
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


def get_class_based_data(df, class_id):
    '''
    Random generate equally sized dataset for binary classifiers of specified class
    by limiting the generated dataset size to minimum across all classes in dataset.
    '''
    size_limit = df['y'].value_counts().min()
    class_df = df.loc[df['y'] == class_id]
    non_class_df = df.loc[df['y'] != class_id]
    max_size = df['y'].value_counts().min()
    class_df = class_df.sample(n=max_size)
    non_class_df = non_class_df.sample(n=max_size)
    return pd.concat([class_df, non_class_df])
