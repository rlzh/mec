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
from utils import get_within_class_cos_similarity
from utils import get_between_class_cos_similarity
from utils import get_shared_words
from numpy import linalg as LA

def gen_word_cloud_grid(name, df, vectorizer, top_n_generator, n=10, order=-1, random_state=None, print_=False):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        '{}: uni-grams per class'.format(name))
    top_n_list = [None] * 4
    if print_:
        print("Generating word clouds for {} using {}...".format(name, type(vectorizer)))
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
        ax.set_title('Angry')
        if i == 1:
            ax = axs[0, 1]
            ax.set_title('Happy')
        elif i == 3:
            ax = axs[1, 0]
            ax.set_title('Sad')
        elif i == 4:
            ax = axs[1, 1]
            ax.set_title('Relaxed')
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


def calc_train_data_cos_sim(name, df, vectorizer, random_state=None, print_=False):
    between_total = 0
    pos_within_total = 0
    neg_within_total = 0
    for i in df.y.unique():
        class_df = get_class_based_data(
            df,
            i,
            random_state=random_state,
        )
        # calc positive within-class cosine similarity
        pos_cos_sim = get_within_class_cos_similarity(class_df, vectorizer, 1)
        print(pos_cos_sim.shape)
        # calc negative within-class cosine similarity
        neg_cos_sim = get_within_class_cos_similarity(class_df, vectorizer, -1)
        print(neg_cos_sim.shape)
        # calc positive negative between-class cosine similarity
        between_cos_sim = get_between_class_cos_similarity(
            class_df, vectorizer, 1, -1)
        
        if print_:
            print("{} Class {}: + cosine sim norm = {}".format(name,
                                                              i, LA.norm(pos_cos_sim)))
            print("{} Class {}: - cosine sim norm = {}".format(name,
                                                            i, LA.norm(neg_cos_sim)))
            print("{} Class {}: + vs - cosine sim norm = {}".format(name,
                                                                i, LA.norm(between_cos_sim)))
        pos_within_total += LA.norm(pos_cos_sim)
        neg_within_total += LA.norm(neg_cos_sim)
        between_total += LA.norm(between_cos_sim)
    if print_:
        average_between = between_total / float(len(df.y.unique()))
        average_pos_within = pos_within_total / len(df.y.unique())
        average_neg_within = neg_within_total / len(df.y.unique())
        print("{}: average + cosine sim norm = {}".format(name, average_pos_within))
        print("{}: average - cosine sim norm = {}".format(name, average_neg_within))
        print("{}: average + vs - cosine sim norm = {}".format(name, average_between))
        print()


def calc_class_cos_sim(name, df, vectorizer, print_=False):
    between_total = 0
    within_total = 0
    classes = [i + 1 for i in range(len(df.y.unique()))]
    count = 0
    within_log = []
    between_log = []
    for i in range(len(classes)):
        # calc within-class cosine similarity
        within_cos_sim = get_within_class_cos_similarity(
            df, vectorizer, classes[i])
        within_total += LA.norm(within_cos_sim)
        within_log.append("{} Class {}: within class cosine sim norm = {}".format(
            name, classes[i], LA.norm(within_cos_sim)))
        for j in range(i + 1, len(classes)):
            # calc between-class cosine similarity
            between_cos_sim = get_between_class_cos_similarity(
                df, vectorizer, classes[i], classes[j])
            between_total += LA.norm(between_cos_sim)
            between_log.append("{}: Class {} vs {} cosine sim norm: {}".format(
                name, classes[i], classes[j], LA.norm(between_cos_sim)))
            count += 1
    between_average = between_total / count
    within_average = within_total / len(classes)
    if print_:
        for l in within_log:
            print(l)
        print("{}: average within cosine sim norm: {}".format(name, within_average))
        for l in between_log:
            print(l)
        print("{}: average between cosine sim norm: {}".format(
            name, between_average))
        print()


def main(*args):

    # load stop words
    stop_words = get_stop_words()

    plot = const.PLOT_DEFAULT
    print_ = const.PRINT_DEFAULT
    max_features = 5000
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
        elif k == 'use_stop_words':
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
        print("Even distribution dataset: {}".format(even_distrib))
        print("Stop words: {}".format(stop_words == None))
        print("Max features: {}".format(max_features))
        print("Random state: {}".format(random_state))
        print("Gen wordcloud: {}".format(wordcloud_))
        print("Wordcloud top N: {}".format(wordcloud_n))
        print("Wordcloud order: {}".format(order))
        print("Calc cos sim: {}".format(cos_sim))
        print("Plot: {}".format(plot))
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
        ngram_range=(1, 1),
    )
    count = CountVectorizer(
        stop_words=stop_words,
        max_df=0.25,
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
        deezer_tfidf_shared, deezer_tfidf_unique = get_shared_words(deezer_top_n)

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
        # Cosine similiarity of training data
        calc_train_data_cos_sim(
            "Spotify",
            clean_spotify_df,
            tfidf,
            print_=print_,
            random_state=random_state,
            
        )
        calc_train_data_cos_sim(
            "Deezer",
            clean_deezer_df,
            tfidf,
            print_=print_,
            random_state=random_state,
        )

        # Cosine similarity of classes in overall dataset
        calc_class_cos_sim(
            "Spotify",
            clean_spotify_df,
            tfidf,
            print_=print_,
        )
        calc_class_cos_sim(
            "Deezer",
            clean_deezer_df,
            tfidf,
            print_=print_
        )

    if plot:
        plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])
