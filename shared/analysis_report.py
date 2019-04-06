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
from utils import get_top_total_ngrams
from utils import get_class_based_data
from utils import get_stop_words
from utils import get_shared_words
from numpy import linalg as LA


def gen_word_cloud_grid(name, df, vectorizer, n=10, order=-1, random_state=None, print_=False):
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
            include_other_classes=True,
            limit_size=False,
        )

        vectorized = vectorizer.fit_transform(class_df.lyrics.values)
        # if print_:
            # print("{} Class {}: feature size={}".format(
                # name, i, vectorized.shape))
        top_n = get_top_total_ngrams(
            vectorized,
            vectorizer,
            n=n,
            order=order,
        )
        wordcloud = WordCloud().generate_from_frequencies(frequencies=top_n)
        ax = axs[0, 0]

        if i == 1:
            ax = axs[0, 1]
            ax.set_title('Happy')
        elif i == 2:
            ax = axs[0, 0]
            ax.set_title('Angry')
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
    random_state = const.RANDOM_STATE_DEFAULT
    order = -1  # default descending order
    wordcloud_n = None
    wordcloud_ = False
    cos_sim = False
    even_distrib = const.EVEN_DISTRIB_DEFAULT
    plt.rcParams.update({'font.size': const.FONT_SIZE_DEFAULT})
    pre_vec = False
    limit_size = False
    min_df = 1
    max_df = 1.0
    param_compare = False

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
        elif k == 'stop_words':
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
        elif k == 'pre_vec':
            pre_vec = utils.str_to_bool(v)
        elif k == 'limit_size':
            limit_size = utils.str_to_bool(v)
        elif k == 'min_df':
            min_df = int(v)
        elif k == 'max_df':
            max_df = float(v)
            if max_df > 1:
                max_df = int(max_df)
        elif k == 'param_compare':
            param_compare = utils.str_to_bool(v)
        else:
            print("Unknown param: {}".format(k))

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
        print("param_compare: {}".format(param_compare))
        print("pre_vec: {}".format(pre_vec))
        print("limit_size: {}".format(limit_size))
        print("min_df: {}".format(min_df))
        print("max_df: {}".format(max_df))
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

    datasets = [
        (const.SPOTIFY, clean_spotify_df),
        (const.DEEZER, clean_deezer_df),
    ]
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        binary=True,
    )

    # word clouds
    if wordcloud_:
        top_n = gen_word_cloud_grid(
            const.SPOTIFY,
            clean_spotify_df,
            vectorizer=vectorizer,
            n=wordcloud_n,
            order=order,
            random_state=random_state,
            print_=print_
        )
        spotify_shared, spotify_unique = get_shared_words(top_n)

        top_n = gen_word_cloud_grid(
            const.DEEZER,
            clean_deezer_df,
            vectorizer=vectorizer,
            n=wordcloud_n,
            order=order,
            random_state=random_state,
            print_=print_
        )
        deezer_shared, deezer_unique = get_shared_words(top_n)

        if print_:
            print()
            print("Spotify: count shared={}".format(
                len(spotify_shared)/len(spotify_unique)))
            print("Deezer: count shared={}".format(
                len(deezer_shared)/len(deezer_unique)))
            print()

    # cosine similarity
    if cos_sim: 
        for name, dataset in datasets:
            if pre_vec:
                dataset = utils.get_vectorized_df(dataset, vectorizer)

            print("{} class data similarity analysis...".format(name))
            for i in dataset.y.unique():
                class_df = utils.get_class_based_data(
                    dataset,
                    i,
                    random_state=random_state,
                    include_other_classes=True,
                    even_distrib=False,
                    limit_size=limit_size,
                    print_=True,
                )
                if pre_vec == False:
                    class_df = utils.get_vectorized_df(class_df, vectorizer)
                pos_df = utils.get_class_based_data(class_df, 1)
                pos_df.pop('y')
                ave_pos = utils.get_average_cos_sim(pos_df.values)
                neg_df = utils.get_class_based_data(class_df, -1.0)
                neg_df.pop('y')
                ave_neg = utils.get_average_cos_sim(neg_df.values)
                ave_between = utils.get_average_cos_sim(
                    pos_df.values, neg_df.values)
                print("class {}".format(i))
                print("data shape: {}".format(class_df.shape))
                print("average positive cosine similarity: {}".format(ave_pos))
                print("average negative cosine similarity: {}".format(ave_neg))
                print("average between cosine similarity: {}".format(ave_between))
                print("(pos - between )+ (neg - between) percentage = {} ".format(
                    (ave_pos - ave_between) / ave_pos + (ave_neg - ave_between)  / ave_neg
                ))
                print()

    if param_compare:
        # min_df vs pos_sim, neg_sim, between_sim
        params_grid = {
            'min_df': [i for i in range(1, 15)],
            'max_df': np.arange(0.1, 1.0, 0.1),
        }

        for name, dataset in datasets:    
            for i in dataset.y.unique():
                df = utils.get_class_based_data(
                    dataset,
                    i,
                    random_state=random_state,
                    include_other_classes=True,
                    even_distrib=False,
                    limit_size=limit_size,
                )
                for p, v in params_grid.items():
                    print("Comparing cosine similarity vs {} for {} Class {} data...".format(p, name, i))
                    vectorizer = CountVectorizer(
                        stop_words=stop_words,
                        ngram_range=(1, 1),
                        min_df=min_df,
                        max_df=max_df,
                        max_features=max_features,
                        binary=True,
                    )
                    pos_sim = []
                    neg_sim = []
                    between_sim = []
                    diff = []
                    for j in range(len(v)):
                        vectorizer.set_params(**{p: v[j]})
                        class_df = utils.get_vectorized_df(df, vectorizer)
                        pos_df = utils.get_class_based_data(class_df, 1)
                        pos_df.pop('y')
                        ave_pos = utils.get_average_cos_sim(pos_df.values)
                        neg_df = utils.get_class_based_data(class_df, -1.0)
                        neg_df.pop('y')
                        ave_neg = utils.get_average_cos_sim(neg_df.values)
                        ave_between = utils.get_average_cos_sim(
                            pos_df.values, neg_df.values)
                        pos_sim.append(ave_pos)
                        neg_sim.append(ave_neg)
                        between_sim.append(ave_between)
                        diff.append((ave_pos - ave_between)/ave_pos + (ave_neg - ave_between)/ave_neg)
                    
                    plt.figure()
                    plt.title("{} Class {}: {} vs cosine similarity".format(name,i, p))
                    pos_sim = np.array(list(zip(v, pos_sim)))
                    neg_sim = np.array(list(zip(v, neg_sim)))
                    between_sim = np.array(list(zip(v, between_sim)))
                    diff = np.array(list(zip(v, diff)))
                    plt.plot(pos_sim[:, 0], pos_sim[:, 1], label='pos sim')
                    plt.plot(neg_sim[:, 0], neg_sim[:, 1], label='neg sim')
                    plt.plot(between_sim[:, 0], between_sim[:, 1], label='between sim')
                    plt.plot(diff[:, 0], diff[:, 1], label='sim difference (%)')
                    plt.xlabel(p)
                    plt.legend()            

    # grid search eval
    if plot:
        plt.draw()
        plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])
