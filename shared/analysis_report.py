import pandas as pd
import numpy as np
import const
import nltk
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
from utils import get_within_data_cos_similarity
from utils import get_between_class_cos_similarity
from utils import get_shared_ngrams


# load stop words
stop_words = get_stop_words()


def gen_word_cloud_grid(name, df, vectorizer, top_n_generator, n=10, order=-1, print_=False):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        '{}: uni-grams per class'.format(name))
    top_n_list = [None] * 4
    for i in df.y.unique():
        class_df = get_class_based_data(
            df,
            i,
            random_state=np.random.seed(0),
            include_other_classes=False,
            limit_size=False,
        )
        vectorized = vectorizer.fit_transform(class_df.lyrics.values)
        if print_:
            print("Class {}: feature size={}".format(i, vectorized.shape))
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
    return np.array(top_n_list)


def gen_within_data_cos_sim_heapmaps(name, df, max_features=1000, print_=False):
    print("Max features: {}".format(max_features))
    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        ngram_range=(1, 1),
    )
    total = 0
    for i in df.y.unique():
        cos_sim = get_within_data_cos_similarity(df, tfidf, i)
        if print_:
            print("{}: Class {} cosine similarity sum: {}".format(
                name, i, cos_sim.sum()))
        total += cos_sim.sum()
    average = total/len(df.y.unique())
    if print_:
        print("{}: average cosine similiarity: {}".format(name, average))


def gen_between_class_cos_sim_heapmaps(name, df, max_features=1000, print_=False):
    print("Max features: {}".format(max_features))
    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        ngram_range=(1, 1),
    )
    total = 0
    classes = [i + 1 for i in range(len(df.y.unique()))]
    count = 0
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            cos_sim = get_between_class_cos_similarity(
                df, tfidf, classes[i], classes[j])
            if print_:
                print("{}: Class {} vs {} cosine similarity: {}".format(
                    name, classes[i], classes[j], cos_sim.sum()))
            total += cos_sim.sum()
            count += 1
    average = total/count
    if print_:
        print("{}: average cosine similarity: {}".format(name, average))


gen_spotify_df = pd.read_csv(const.GEN_SPOTIFY)
clean_spotify_df = pd.read_csv(const.CLEAN_SPOTIFY)

gen_deezer_df = pd.read_csv(const.GEN_DEEZER)
clean_deezer_df = pd.read_csv(const.CLEAN_DEEZER)

show = False
print_ = True

# word clouds
n = None
tfidf = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 1),
)
count = CountVectorizer(
    stop_words=stop_words,
    max_df=0.25,
    ngram_range=(1, 1),
)
spotify_top_n = gen_word_cloud_grid(
    'Spotify',
    clean_spotify_df,
    vectorizer=tfidf,
    top_n_generator=get_top_idf_ngrams,
    n=n,
    order=1,
    print_=print_
)
spotify_tfidf_shared, spotify_tfidf_unique = get_shared_ngrams(spotify_top_n)

deezer_top_n = gen_word_cloud_grid(
    'Deezer',
    clean_deezer_df,
    vectorizer=tfidf,
    top_n_generator=get_top_idf_ngrams,
    n=n,
    order=1,
    print_=print_
)
deezer_tfidf_shared, deezer_tfidf_unique = get_shared_ngrams(deezer_top_n)


top_n = gen_word_cloud_grid(
    'Spotify',
    clean_spotify_df,
    vectorizer=count,
    top_n_generator=get_top_total_ngrams,
    n=n,
    order=-1,
    print_=print_
)
spotify_count_shared, spotify_count_unique = get_shared_ngrams(top_n)

top_n = gen_word_cloud_grid(
    'Deezer',
    clean_deezer_df,
    vectorizer=count,
    top_n_generator=get_top_total_ngrams,
    n=n,
    order=-1,
    print_=print_
)
deezer_count_shared, deezer_count_unique = get_shared_ngrams(top_n)

print()
print("Spotify: tfidf shared={}".format(
    len(spotify_tfidf_shared)/len(spotify_tfidf_unique)))
print("Deezer: tfidf shared={}".format(
    len(deezer_tfidf_shared)/len(deezer_tfidf_unique)))
print("Spotify: count shared={}".format(
    len(spotify_count_shared)/len(spotify_count_unique)))
print("Deezer: count shared={}".format(
    len(deezer_count_shared)/len(deezer_count_unique)))
print()

# vectorized = count.fit_transform(clean_spotify_df.lyrics.values)
# gen_word_cloud(get_top_total_ngrams(
#     vectorized,
#     count,
#     n=n
# ))

# vectorized = count.fit_transform(clean_deezer_df.lyrics.values)
# gen_word_cloud(get_top_total_ngrams(
#     vectorized,
#     count,
#     n=n
# ))


# Cosine similiarity within data
max_features = 5000
order = 1
gen_within_data_cos_sim_heapmaps(
    "Spotify",
    clean_spotify_df,
    max_features=max_features,
    print_=print_,
)
gen_within_data_cos_sim_heapmaps(
    "Deezer",
    clean_deezer_df,
    max_features=max_features,
    print_=print_,
)

# Cosine similarity between class
max_features = 5000
gen_between_class_cos_sim_heapmaps(
    "Spotify",
    clean_spotify_df,
    max_features=max_features,
    print_=print_
)
gen_between_class_cos_sim_heapmaps(
    "Deezer",
    clean_deezer_df,
    max_features=max_features,
    print_=print_
)

if show:
    plt.show()
