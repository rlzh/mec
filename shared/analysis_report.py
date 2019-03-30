import pandas as pd
import numpy as np
import const
import utils
import nltk
import sys
import seaborn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import paired_euclidean_distances
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from utils import get_word_counts
from utils import get_unique_counts
from utils import get_emotion_counts
from utils import get_top_idf_ngrams
from utils import get_top_total_ngrams
from utils import get_class_based_data
from utils import get_stop_words
from utils import get_shared_words
from numpy import linalg as LA


def gen_word_cloud_grid(name, df, vectorizer, top_n_generator, n=10, order=-1, random_state=None, print_=False):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        '{}: uni-grams per class'.format(name))
    top_n_list = [None] * len(df.y.unique())
    if print_:
        print("Generating word clouds for {} using {}...".format(
            name, type(vectorizer)))
    for i in df.y.unique():
        class_df = get_class_based_data(
            df,
            i,
            random_state=random_state,
            include_other_classes=False,
            limit_size=False,
        )

        vectorized = vectorizer.fit_transform(class_df.lyrics.values)
        # if print_:
            # print("{} Class {}: feature size={}".format(
                # name, i, vectorized.shape))
        top_n = top_n_generator(
            vectorized,
            vectorizer,
            n=n,
            order=order,
        )
        # if print_:
        # print("Top {} for Class {}\n  {}".format(n, i, top_n))
        wordcloud = WordCloud().generate_from_frequencies(frequencies=top_n)
        ax = axs[0, 0]
        # ax.set_title('Angry')
        if i == 1:
            ax = axs[0, 1]
            # ax.set_title('Happy')
        elif i == 3:
            ax = axs[1, 0]
            # ax.set_title('Sad')
        elif i == 4:
            ax = axs[1, 1]
            # ax.set_title('Relaxed')
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        keys = []
        for key in top_n:
            keys.append(key)
        top_n_list[int(i)-1] = keys
    fig.tight_layout()
    if print_:
        print()

    return np.array(top_n_list)

def calc_class_cos_sim(name, df, vectorizer, print_=False):
    between_total = 0
    classes = [i + 1 for i in range(len(df.y.unique()))]
    count = 0
    between_log = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            # calc between-class cosine similarity
            between_cos_sim = get_between_class_cos_similarity(
                df, vectorizer, classes[i], classes[j])
            between_total += (between_cos_sim.sum()) / \
                between_cos_sim.shape[0]**2
            between_log.append("{}: Class {} vs {} cosine sim norm: {}".format(
                name, classes[i], classes[j], (between_cos_sim.sum())/between_cos_sim.shape[0]**2))
            count += 1
    between_average = between_total / count
    if print_:
        for l in between_log:
            print(l)
        print("{}: average between cosine sim norm: {}".format(
            name, between_average))
        print()
    return between_average


def main(*args):

    # load stop words
    stop_words = get_stop_words()

    plot = const.PLOT_DEFAULT
    print_ = const.PRINT_DEFAULT
    max_features = None
    random_state = None
    order = -1  # default descending order
    wordcloud_n = None
    wordcloud_ = True
    cos_sim = True
    even_distrib = const.EVEN_DISTRIB_DEFAULT
    plt.rcParams.update({'font.size': const.FONT_SIZE_DEFAULT})

    # print command line arguments
    for arg in args:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if k == 'plot':
            plot = utils.str_to_bool(v)
        elif k == 'print':
            print_ = utils.str_to_bool(v)
        elif k == 'max_features':
            max_features = int(v)
        elif k == 'stop_words·':
            if utils.str_to_bool(v) == False:
                stop_words = None
        elif k == 'random_state':
            random_state = int(v)
        elif k == 'order':
            order = int(v)
        elif k == 'wordcloud':
            wordcloud_ = utils.str_to_bool(v)
        elif k == 'wordcloud_n':
            wordcloud_n = int(v)
        elif k == 'cos_sim':
            cos_sim = utils.str_to_bool(v)
        elif k == 'font_size':
            plt.rcParams.update({'font.size': int(v)})
        elif k == 'even_distrib':
            even_distrib = utils.str_to_bool(v)

    if print_:
        print()
        print("-- Analysis config --")
        print("even_distrib: {}".format(even_distrib))
        print("stop_words: {}".format(stop_words != None))
        print("max_features: {}".format(max_features))
        print("random_state: {}".format(random_state))
        print("wordcloud: {}".format(wordcloud_))
        print("wordcloud_n: {}".format(wordcloud_n))
        print("order: {}".format(order))
        print("cos_sim: {}".format(cos_sim))
        print("plot: {}".format(plot))
        print("--------------------")
        print()

    gen_spotify_df = pd.read_csv(const.GEN_SPOTIFY)
    clean_spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)
    if even_distrib == False:
        clean_spotify_df = pd.read_csv(const.CLEAN_UNEVEN_SPOTIFY)

    gen_deezer_df = pd.read_csv(const.GEN_DEEZER)
    clean_deezer_df = pd.read_csv(const.CLEAN_DEEZER)
    if even_distrib == False:
        clean_deezer_df = pd.read_csv(const.CLEAN_UNEVEN_DEEZER)

    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        ngram_range=(1, 1),
    )
    count = CountVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        ngram_range=(1, 1),
    )

    # word clouds
    if wordcloud_:
        spotify_top_n = gen_word_cloud_grid(
            'Spotify',
            clean_spotify_df,
            vectorizer=tfidf,
            top_n_generator=get_top_idf_ngrams,
            n=wordcloud_n,
            order=order,
            random_state=random_state,
            print_=print_
        )
        spotify_tfidf_shared, spotify_tfidf_unique = get_shared_words(
            spotify_top_n)

        deezer_top_n = gen_word_cloud_grid(
            'Deezer',
            clean_deezer_df,
            vectorizer=tfidf,
            top_n_generator=get_top_idf_ngrams,
            n=wordcloud_n,
            order=order,
            random_state=random_state,
            print_=print_
        )
        deezer_tfidf_shared, deezer_tfidf_unique = get_shared_words(
            deezer_top_n)

        top_n = gen_word_cloud_grid(
            'Spotify',
            clean_spotify_df,
            vectorizer=count,
            top_n_generator=get_top_total_ngrams,
            n=wordcloud_n,
            order=order,
            random_state=random_state,
            print_=print_
        )
        spotify_count_shared, spotify_count_unique = get_shared_words(top_n)

        top_n = gen_word_cloud_grid(
            'Deezer',
            clean_deezer_df,
            vectorizer=count,
            top_n_generator=get_top_total_ngrams,
            n=wordcloud_n,
            order=order,
            random_state=random_state,
            print_=print_
        )
        deezer_count_shared, deezer_count_unique = get_shared_words(top_n)

        if print_:
            print()
            print("Spotify: tfidf shared={}".format(
                len(spotify_tfidf_shared) / len(spotify_tfidf_unique)))
            print("Spotify: count shared={}".format(
                len(spotify_count_shared)/len(spotify_count_unique)))
            print("Deezer: tfidf shared={}".format(
                len(deezer_tfidf_shared)/len(deezer_tfidf_unique)))
            print("Deezer: count shared={}".format(
                len(deezer_count_shared)/len(deezer_count_unique)))
            print()

    if cos_sim:
        vectorizer = CountVectorizer(
            # stop_words=stop_words,
            ngram_range=(1, 1),
            # min_df=5,
            max_df=1.0,
            max_features=max_features
        )
        for name, dataset in [('spotify', clean_spotify_df), ('deezer',clean_deezer_df)]:
            vectorized_df = utils.get_vectorized_df(
                clean_spotify_df, vectorizer)
            print("{} class data similarity analysis...".format(name))
            for i in vectorized_df.y.unique():
                class_df = utils.get_class_based_data(
                    vectorized_df,
                    i,
                    random_state=random_state,
                    include_other_classes=True,
                )
                pos_df = utils.get_class_based_data(class_df,1,)
                pos_df.pop('y')
                ave_pos = utils.get_average_cos_sim(pos_df.values)
                neg_df = utils.get_class_based_data(class_df,-1,)
                neg_df.pop('y')
                ave_neg = utils.get_average_cos_sim(neg_df.values)
                print("class {}".format(i))
                print("data shape: {}".format(class_df.shape))
                print("average positive cosine similarity: {}".format(ave_pos))
                print("average negative cosine similarity: {}".format(ave_neg))
                ave_between = utils.get_average_cos_sim(pos_df.values, neg_df.values)
                print("average between cosine similarity: {}".format(ave_between))
                print()
    if plot:
        plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])
