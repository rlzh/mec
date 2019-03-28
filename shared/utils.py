import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def str_to_bool(s):
    return True if s.lower() == 'true' or s == str(1) or s.lower() == 'on' else False


def get_stop_words():
    ''' Builds and returns stopswords list from nltk and local text file '''
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    f = open("stopwords.txt", 'r')
    for l in f:
        if len(l.strip()) > 0:
            stop_words.add(l.strip())
    return stop_words


def print_song(df, i):
    ''' Print song from dataframe at index i '''
    print("Song: {}".format(df.song[i].value))
    print("Artist: {}".format(df.artist[i].value))
    print("Lyrics: {}".format(df.lyrics[i].value))


def get_word_counts(lyrics, print_=False):
    ''' Calculates the word count for each lyric and returns result as array '''
    word_count = np.empty(shape=(len(lyrics),))
    for i in range(len(lyrics)):
        if lyrics[i] != np.nan:
            words = str(lyrics[i]).split(' ')
            word_count[i] = len(words)
        else:
            word_count[i] = 0
    if print_ == True:
        print("Word count info")
        print("min word count: {} (id: {})".format(
            np.min(word_count), np.argmin(word_count)))
        print("max word count: {} (id: {})".format(
            np.max(word_count), np.argmax(word_count)))
        print("ave word count: {}".format(np.mean(word_count)))
        print()
    return word_count


def get_unique_counts(lyrics, print_=False):
    ''' 
    Calculates the unique word count for each lyric and returns 
    result as array 
    '''
    unique_count = np.empty(shape=(len(lyrics),))
    for i in range(len(lyrics)):
        if lyrics[i] != np.nan:
            words = str(lyrics[i]).split(' ')
            s = set()
            for w in words:
                if w not in s:
                    s.add(w)
                    unique_count[i] += 1
        else:
            unique_count[i] = 0

    if print_ == True:
        print("Unique count info")
        print("min unique count: {} (id: {})".format(
            np.min(unique_count), np.argmin(unique_count)))
        print("max unique count: {} (id: {})".format(
            np.max(unique_count), np.argmax(unique_count)))
        print("ave unique count: {}".format(np.mean(unique_count)))
        print()

    return unique_count


def get_emotion_counts(df, print_=False):
    ''' Returns the emotion class distribution as pandas DF '''
    class_distrib = df['y'].value_counts()
    if print_ == True:
        print("Emotion info")
        print("valence range: [{},{}]".format(
            df.valence.min(), df.valence.max()))
        print("arousal range: [{},{}]".format(
            df.arousal.min(), df.arousal.max()))
        print(class_distrib)
        print()
    return class_distrib


def get_top_idf_ngrams(vectorized, vectorizer, n=10, order=-1):
    ''' 
    Returns the top ngrams according to IDF as dictionary
     (key=ngram, value=IDF score)
    '''
    feature_names = np.array(vectorizer.get_feature_names())
    vectorized_sorted = np.sort(vectorizer.idf_.flatten())[::order]
    vectorized_sorting = np.argsort(vectorizer.idf_).flatten()[::order]
    top_n = feature_names[vectorized_sorting][:n]
    top_n_scores = vectorized_sorted[:n].reshape(-1, )
    # top_n_scores = np.add(top_n_scores, 1.0).reshape(-1,)
    top_n = dict(zip(top_n, top_n_scores))
    return top_n


def get_top_total_ngrams(vectorized, vectorizer, n=10, order=-1):
    ''' 
    Returns the top ngrams according to sum of columns as 
    dictionary (key=ngram, value=sum of column)
    '''
    feature_names = np.array(vectorizer.get_feature_names())
    vectorized_sorted = np.sort(vectorized.toarray().sum(axis=0))[::order]
    vectorized_sorting = np.argsort(vectorized.toarray().sum(axis=0))[::order]
    top_n = feature_names[vectorized_sorting][:n]
    top_n_scores = vectorized_sorted[:n].reshape(-1,)
    # top_n_scores = np.add(top_n_scores, 1.0).reshape(-1,)
    top_n = dict(zip(top_n, top_n_scores))
    return top_n


def get_within_data_cos_similarity(df, vectorizer, class_id, limit_size=True, even_distrib=True):
    ''' 
    Get within-data pairwise cosine similarity of positive and 
    negative data samples for specified class 
    '''
    class_df = get_class_based_data(
        df,
        class_id,
        include_other_classes=True,
        limit_size=limit_size,
        even_distrib=even_distrib,
    )
    vectorized = vectorizer.fit_transform(class_df.lyrics.values).toarray()
    y_col = class_df.y.values
    pos_indices = np.argwhere(y_col == 1)
    neg_indices = np.argwhere(y_col == -1)
    pos = np.take(vectorized, pos_indices)
    neg = np.take(vectorized, neg_indices)
    cos_sim = cosine_similarity(pos, neg)
    return cos_sim


def get_within_class_cos_similarity(df, vectorizer, class_id):
    class_df = get_class_based_data(
        df,
        class_id,
        include_other_classes=False,
        limit_size=False,
    )
    vectorized = vectorizer.fit_transform(class_df.lyrics.values).toarray()
    cos_sim = cosine_similarity(vectorized, vectorized)
    return cos_sim


def get_between_class_cos_similarity(df, vectorizer, primary_class, secondary_class):
    '''
    Get between-class pairwise cosine similarity of primary class and
    secondary class
    '''
    primary_df = get_class_based_data(
        df,
        primary_class,
        include_other_classes=False,
        limit_size=False,
    )
    secondary_df = get_class_based_data(
        df,
        secondary_class,
        include_other_classes=False,
        limit_size=False,
    )
    primary_vec = vectorizer.fit_transform(primary_df.lyrics.values)
    secondary_vec = vectorizer.fit_transform(secondary_df.lyrics.values)
    if primary_vec.shape[1] != secondary_vec.shape[1]:
        vectorizer.max_features = min(
            primary_vec.shape[1],
            secondary_vec.shape[1]
        )
        primary_vec = vectorizer.fit_transform(primary_df.lyrics.values)
        secondary_vec = vectorizer.fit_transform(secondary_df.lyrics.values)
    # calculate pairwise cosine similarity between all class feature matrices
    cos_sim = cosine_similarity(primary_vec, secondary_vec)
    return cos_sim


def get_shared_words(lyrics):
    ''' 
    Return shared words & unique ngrams over all classes in lyrics
    '''
    shared_words = set()
    unique_words = set()
    hashed_data = []
    for row in lyrics:
        hashed_data.append(set(row))
    for row in lyrics:
        for word in row:
            if word in shared_words:
                continue
            unique_words.add(word)
            is_shared = True
            for other_row in hashed_data:
                if word not in other_row:
                    is_shared = False
                    break
            if is_shared:
                shared_words.add(word)
    return shared_words, unique_words


def get_class_based_data(df, class_id, random_state=None, include_other_classes=True, limit_size=True, even_distrib=True):
    '''
    Random generate equally sized dataset for binary classifiers of specified class
    by limiting the generated dataset size to minimum across all classes in dataset.
    '''
    max_size = df['y'].value_counts().min()
    pos_df = df.loc[df['y'] == class_id]
    neg_df = df.loc[df['y'] != class_id]

    if limit_size == False:
        # no limit so take max between pos & neg df
        max_size = min(len(neg_df), len(pos_df))

    if include_other_classes:
        if even_distrib:
            # even distribution of negative class data between negative classes
            required_size = int(max_size / 3)
            min_size = df['y'].value_counts().min()
            if required_size > min_size:
                # not all classes can meet required size - need to update max_size, required_size
                required_size = min_size

            max_size = required_size * 3
            neg_df = pd.DataFrame([], columns=df.columns)
            for i in df.y.unique():
                if i != class_id:
                    rows = df.loc[df['y'] == i].sample(
                        n=required_size,
                        random_state=random_state
                    )
                    neg_df = pd.concat(
                        [neg_df, rows],
                        ignore_index=True
                    )
        else:
            neg_df = neg_df.sample(n=max_size, random_state=random_state)

    pos_df = pos_df.sample(
        n=max_size,
        random_state=random_state
    )
    pos_df.y = np.full(pos_df.y.shape, 1)
    result = pos_df
    if include_other_classes:
        # print(neg_df.y.value_counts())
        neg_df.y = np.full(neg_df.y.shape, -1)
        result = pd.concat([pos_df, neg_df])
    result.columns = df.columns
    return result


def get_even_distrib_split(df, test_size=0.1, random_state=None):
    ''' 
    Returns split of dataset with even distribution of
    each class in train and test sets. Returns as pandas DF
    '''
    train = np.empty([0, df.shape[1]])
    test = np.empty([0, df.shape[1]])
    train_size = 1 - test_size
    for i in df.y.unique():
        class_data = df.loc[df.y == i].values
        if random_state != None:
            np.random.seed(random_state)
        # shuffle values
        np.random.shuffle(class_data)
        split_index = int(len(class_data) * train_size)
        # split into training and test sets
        class_train = class_data[:split_index, :]
        class_test = class_data[split_index:, :]
        # concat to existing data
        train = np.concatenate((train, class_train), axis=0)
        test = np.concatenate((test, class_test), axis=0)
    if random_state != None:
        np.random.seed(random_state)
    # shuffle dataset order
    np.random.shuffle(train)
    np.random.shuffle(test)
    # return as pandas DF
    train_df = pd.DataFrame(train, columns=df.columns)
    test_df = pd.DataFrame(test, columns=df.columns)
    return train_df, test_df
