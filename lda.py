#!/usr/bin/env python
# coding: utf-8

# In[3]:


from shared import const
from shared import utils
import numpy as np
import pandas as pd
import re

# Gensim
import gensim
from gensim import models
import gensim.corpora as corpora
from gensim.utils import simple_preprocess, lemmatize
from gensim.models import CoherenceModel, LdaModel, LdaMulticore

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

#Choose Random forest for multiclass classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score


# In[4]:


#prepare stopwords
from nltk.corpus import stopwords

stop_words = stopwords.words("english")

#for the purpose of emotion classification, we trim the words by extending stopwords:
stop_words.extend(["from", "re", "use", "day", "let", "get","say", "know", "think", "love"])


# In[5]:


#preproccesing

#remove words fewer than 3 characters
def remove_words(list2d_words, threshold = 3):
    '''
    return a 2d list of words
    '''
    words_more_than_3chars = []
    for i in range(len(list2d_words)):
        proc_sentence = []
        for j in range(len(list2d_words[i])):
            if len(list2d_words[i][j]) >= threshold:
                proc_sentence.append(list2d_words[i][j])
        words_more_than_3chars.append(proc_sentence)
    return words_more_than_3chars
        

#remove stopwords
#simple_preprocess will remove punctuations and unnecessary characters altogether
def remove_stopwords(list2d_words):
    return [[word for word in simple_preprocess(" ".join(doc)) if word not in stop_words] for doc in list2d_words]

#lemmatize: word in third person changed to first person, etc, and only keep the nouns, adj, verb, adv
nlp = spacy.load('en', disable = ['parse', 'ner'])

def my_lemmatize(list2d_words, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in list2d_words:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[6]:


#tokenize each document into a list of words, removing punctuations and unnecessary characters altogether.
def sentences_to_words(sentences):
    yield(gensim.utils.simple_preprocess(str(sentences), deacc=True))  # deacc=True removes punctuations

import itertools

def corpus_to_2d_words(corpus):
    word_corpus = []
    for i in range(len(corpus)):
        data_words = list(sentences_to_words(corpus[i]))
        word_corpus = [*word_corpus, *data_words]
    return word_corpus

def words_2d_to_dict(word_2d_list):
    dictionary = corpora.Dictionary(word_2d_list)
    return dictionary

def df_to_lemmatized_words(df):
    words_2d = corpus_to_2d_words(df)
    words_morethan_3chars = remove_words(words_2d)
    words_nostops = remove_stopwords(words_morethan_3chars)
    lemmatized_words = my_lemmatize(words_nostops)
    return lemmatized_words

def lemmatized_to_corpus(words_2d):
    dict_words = words_2d_to_dict(words_2d)
    corpus = [dict_words.doc2bow(simple_preprocess(" ".join(line))) for line in words_2d]
    
def build_LDA_model(dict_words, corpus):
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                          id2word = dict_words,
                                          num_topics = NUM_TOPICS, #tried 3, 4, 5, 6, 10 , 20. seems 3 for coherence is the best, the second is 4
                                          random_state = 100, #the seed 
                                          update_every = 0, #batch mode
                                          chunksize = 1000, #larger the heavier workload for the memory
                                          passes = 10,#how to determine the chunksize and passes?relevant to size of dataset?
                                          alpha = 'auto',
                                          minimum_probability = 0.0,
                                          per_word_topics = False #non-indicative words would be omitted
                                          )
    return lda_model
    
#evaluate by perplexity and coherence
def print_coherence_perplexity(the_model, lemmatized_words, dict_words, corpus, f=None):
    coherence_model = CoherenceModel(model = the_model, texts = lemmatized_words, dictionary = dict_words, coherence = 'c_v')
    coherence_lda = coherence_model.get_coherence()
    #compute model coherence: higher the better:
    print('\nCoherence Score: ', coherence_lda, file = f)
    #compute model perplexity: lower the better:
    print('\nPerplexity: ', the_model.log_perplexity(corpus), file = f)
    
#get the LDA generated topic-document dataset and responding label for classification
def generate_data(model, corpus):
    topics_features = []
    for i in range(len(corpus)):
        row = []
        for j in range(NUM_TOPICS):
            row.append(model.get_document_topics(corpus[i])[j][1])
        topics_features.append(row)
    #generate data for supervised classification
    print("length of topics_features is : {}".format(len(topics_features)))
    gen_df_mid = pd.DataFrame(topics_features)
    return gen_df_mid

def generate_train_test(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test

#z-score normalization
def zscore_normalization(X_train, X_test):
    zscoreScaler = StandardScaler().fit(X_train)
    X_tr_std = zscoreScaler.transform(X_train)
    X_ts_std = zscoreScaler.transform(X_test)
    return X_tr_std, X_ts_std

def RF_classify_score(X_tr_std, X_ts_std, y_train, y_test):
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_tr_std, y_train)
    rf_pred = rf_clf.predict(X_ts_std)
    score = accuracy_score(y_test, rf_pred)
    print(score)
    return score

def complement_classify_score(X_train, X_test, y_train, y_test):
    cnb = ComplementNB(alpha = 0.1)
    cnb.fit(X_train, y_train)
    cnb_pred = cnb.predict(X_test)
    score = accuracy_score(y_test, cnb_pred)
    print(score)
    return score  
    
def randomForest_LDA(X, Y):
    X1_train, X1_test, y1_train, y1_test = generate_train_test(X,Y)
    #z-score normalization
    X1_tr_std, X1_ts_std = zscore_normalization(X1_train, X1_test)
    RF_classify_score(X1_tr_std, X1_ts_std, y1_train, y1_test)
    
def complementNB_LDA(X, Y):
    X_train, X_test, y_train, y_test = generate_train_test(X,Y)
    #X_tr_std, X_ts_std = zscore_normalization(X_train, X_test)
    complement_classify_socre(X_train, X_test, y_train, y_test)


# In[7]:


# Create the TF-IDF model
def build_tfidf_model(corpus):
    tfidf = models.TfidfModel(corpus, smartirs='ntc')
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf


# In[24]:


#use k-fold validation:
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
kFolder = KFold(n_splits = 10, shuffle = True) #10-fold cross-validation

def kfold_classification(X, Y, clf, normalization_func = None, f = None):
   sum_accuracy = []
   for k, (train, test) in enumerate(kFolder.split(X, Y)):
       X_train = np.array(X.iloc[train, :])
       X_test = np.array(X.iloc[test, :])
       y_train = np.array(Y.iloc[train])
       #print(y_train.shape)
       y_test = np.array(Y.iloc[test])
       if normalization_func != None:
           X_train, X_test = normalization_func(X_train, X_test)
       #rf_clf = RandomForestClassifier()
       clf.fit(X_train, y_train)
       rf_pred = clf.predict(X_test)
       accuracy = accuracy_score(y_test, rf_pred)
       sum_accuracy.append(accuracy)
       #print("the {} th fold split got accuracy {}; f1-score: {} ".format(k, accuracy, f1_score(y_test, rf_pred)), file = f)
   k_fold_avg = sum(sum_accuracy)/len(sum_accuracy)
   print("the average of 10 rounds is {}".format(k_fold_avg), file = f)
   return k_fold_avg

#plot topic distribution for each class: add up the sum of portion of each topic
def plot_topics_per_class(df, gen_df_mid, caption):
   pos = df[df['y'] == 1].index.values.astype(int)
   topics = gen_df_mid_1.sum(axis = 0)

   plot_x = [x+1 for x in range(gen_df_mid_1.shape[1])]
   plt.plot(plot_x, topics)
   plt.xlabel("topic ID")
   plt.ylabel("topic portion in the class")
   plt.savefig(caption + '.png')


# In[10]:


#Build LDA model
data_file = [const.CLEAN_UNEVEN_SPOTIFY, const.CLEAN_UNEVEN_DEEZER]
for every in data_file:
    name = str(every).split('/')[-1].split('.')[0]
    f_topic = open('0405_topicno_' + name + '.txt', 'a')
    f_cc = open('0405_coherence_accuracyScore_each' + name +'.txt', 'a')    
    df_full = pd.read_csv(every)
    #print(df_full.shape)        
    df_1 = utils.get_class_based_data(df_full, 1, include_other_classes=True) #
    #print(df_1.head(2))
    print("df_1.y.unique() is ", df_1.y.unique())
    df_2 = utils.get_class_based_data(df_full, 2, include_other_classes=True)
    df_3 = utils.get_class_based_data(df_full, 3, include_other_classes=True)
    df_4 = utils.get_class_based_data(df_full, 4, include_other_classes=True)

    df_1.reset_index()
    df_2.reset_index()
    df_3.reset_index()
    df_4.reset_index()
    Y1 = df_1.iloc[:, -1]
    print("length of y1 is :", len(Y1))
    Y2 = df_2.iloc[:, -1]
    print("length of y2 is : {}".format(len(Y2)))
    Y3 = df_3.iloc[:, -1]
    Y4 = df_4.iloc[:, -1]
    train_dflist_1 = df_1.lyrics[:].tolist()
    #print("length of train_dflist_1 is : {}".format(len(train_dflist_1)))
    train_dflist_2 = df_2.lyrics[:].tolist()
    train_dflist_3 = df_3.lyrics[:].tolist()
    train_dflist_4 = df_4.lyrics[:].tolist()
    
    lemmatized_words_1 = df_to_lemmatized_words(train_dflist_1)
    #print("length of lemmatized_words_1 is : {}".format(len(lemmatized_words_1)))
    lemmatized_words_2 = df_to_lemmatized_words(train_dflist_2)
    lemmatized_words_3 = df_to_lemmatized_words(train_dflist_3)
    lemmatized_words_4 = df_to_lemmatized_words(train_dflist_4)

    #print(lemmatized_words)

    #turn the 2d word list to dict 
    dict_words_1 = words_2d_to_dict(lemmatized_words_1)
    print("length of dict_words_1 is : {}".format(len(dict_words_1)))
    dict_words_2 = words_2d_to_dict(lemmatized_words_2)
    dict_words_3 = words_2d_to_dict(lemmatized_words_3)
    dict_words_4 = words_2d_to_dict(lemmatized_words_4)
    #print(dict_words)

    #there's 2 input for lDA: dictinary and the corpus
    #create bag of words
    corpus_1 =  [dict_words_1.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_1]
    print("length of corpus_1 is : {}".format(len(corpus_1)))
    corpus_2 =  [dict_words_2.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_2]
    corpus_3 =  [dict_words_3.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_3]
    corpus_4 =  [dict_words_4.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_4]
    for i in range(3, 9):
        NUM_TOPICS = i
        lda_model_1 = build_LDA_model(dict_words_1, corpus_1)
        lda_model_2 = build_LDA_model(dict_words_2, corpus_2)

        lda_model_3 = build_LDA_model(dict_words_3, corpus_3)

        lda_model_4 = build_LDA_model(dict_words_4, corpus_4)
        print("---------------------start of topic # {}".format(i), "-------------------", file = f)
        print_coherence_perplexity(lda_model_1, lemmatized_words_1, dict_words_1, corpus_1, file = f_cc)
    
        print_coherence_perplexity(lda_model_2, lemmatized_words_2, dict_words_2, corpus_2, file = f_cc)

        print_coherence_perplexity(lda_model_3, lemmatized_words_3, dict_words_3, corpus_3, file = f_cc)

        print_coherence_perplexity(lda_model_4, lemmatized_words_4, dict_words_4, corpus_4, file = f_cc)

        gen_df_mid_1 = generate_data(lda_model_1, corpus_1)
        print("length of gen_df_mid_1[0] is : {}".format(len(gen_df_mid_1[0])))
        print(gen_df_mid_1.head())
        gen_df_mid_2 = generate_data(lda_model_2, corpus_2)
        gen_df_mid_3 = generate_data(lda_model_3, corpus_3)
        gen_df_mid_4 = generate_data(lda_model_4, corpus_4)
        captions = ['topics of class '+x for x in range(4)]
        plot_topics_per_class(df_1, gen_df_mid_1, captions[0])
        plot_topics_per_class(df_2, gen_df_mid_2, captions[1])
        plot_topics_per_class(df_3, gen_df_mid_3, captions[2])
        plot_topics_per_class(df_4, gen_df_mid_4, captions[3])
    #randomForest_LDA(gen_df_mid_1, df_1.iloc[:TRAIN_SAMPLE, -1])

    #kfold for class 1:
        print("random forest classificaton:", f_cc)
        rf_clf = RandomForestClassifier()
        rf_score_sum = kfold_classification(gen_df_mid_1, Y1, rf_clf, zscore_normalization, f_cc)
    
    #randomForest_LDA(gen_df_mid_2, df_2.iloc[:TRAIN_SAMPLE, -1])
    #kfold for class 2:
    #random forest classificaton:
        rf_clf = RandomForestClassifier()
        rf_score_sum = rf_score_sum + kfold_classification(gen_df_mid_2, Y2, rf_clf, zscore_normalization, f_cc)
    
    #randomForest_LDA(gen_df_mid_3, df_3.iloc[:TRAIN_SAMPLE, -1])
    #kfold for class 3:
    #random forest classificaton:
        rf_clf = RandomForestClassifier()
        rf_score_sum = rf_score_sum + kfold_classification(gen_df_mid_3, Y3, rf_clf, zscore_normalization, f_cc)
    
    #randomForest_LDA(gen_df_mid_4, df_4.iloc[:TRAIN_SAMPLE, -1])
    #kfold for class 4:
    #random forest classificaton:
        rf_clf = RandomForestClassifier()
        rf_score_sum = rf_score_sum + kfold_classification(gen_df_mid_4, Y4, rf_clf, zscore_normalization, f_cc)
        rf_score_avg = rf_score_sum / 4
        print("topics: {}, RF score average of 4 models: {}".format(i, rf_score_avg), file = f)

        print("complement naive bayes classifier:", f_cc)
        cnb_clf = ComplementNB(alpha= 0.1)
        cnb_score_sum = kfold_classification(gen_df_mid_1, Y1, cnb_clf, None, f_cc)
    
    #complement naive bayes classifier:
        cnb_clf = ComplementNB(alpha= 0.1)
        cnb_score_sum = cnb_score_sum + kfold_classification(gen_df_mid_2, Y2, cnb_clf, None, f_cc)

    #complement naive bayes classifier:
        cnb_clf = ComplementNB(alpha= 0.1)
        cnb_score_sum = cnb_score_sum + kfold_classification(gen_df_mid_3, Y3, cnb_clf, None, f_cc)
    
    #complement naive bayes classifier:
        cnb_clf = ComplementNB(alpha= 0.1)
        cnb_score_sum = cnb_score_sum + kfold_classification(gen_df_mid_4, Y4, cnb_clf, None, f_cc)
        cnb_score_avg = cnb_score_sum / 4
        print("Complement NB score average of 4 models: {}".format(cnb_score_avg), file = f)
        
        #kfold for class 1:
        print("KNN classificaton:", f_cc)
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = kfold_classification(gen_df_mid_1, Y1, knn_clf, zscore_normalization, f_cc)
    
        #randomForest_LDA(gen_df_mid_2, df_2.iloc[:TRAIN_SAMPLE, -1])
        #kfold for class 2:
  
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = knn_score_sum + kfold_classification(gen_df_mid_2, Y2, knn_clf, zscore_normalization, f_cc)
    
    
        #kfold for class 3:
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = knn_score_sum + kfold_classification(gen_df_mid_3, Y3, knn_clf, zscore_normalization, f_cc)
    
    
        #kfold for class 4:
        rf_clf = RandomForestClassifier()
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = knn_score_sum + kfold_classification(gen_df_mid_4, Y4, knn_clf, zscore_normalization, f_cc)
    
        knn_score_avg = knn_score_sum / 4
        print("topics: {}, KNN score average of 4 models: {}".format(i, knn_score_avg), file = f)
        print("---------------------end of topic # {}".format(i), "-------------------", file = f)
        f.flush()
    f.close()


# In[ ]:


#find the best k for KNN

#read data
data_file = [const.CLEAN_UNEVEN_SPOTIFY, const.CLEAN_UNEVEN_DEEZER]
for every in data_file:
    df_full = pd.read_csv(every)
    print(df_full.shape)
    df_1 = utils.get_class_based_data(df_full, 1, limit_size = True) 
    df_2 = utils.get_class_based_data(df_full, 2, limit_size = True)
    df_3 = utils.get_class_based_data(df_full, 3, limit_size = True)
    df_4 = utils.get_class_based_data(df_full, 4, limit_size = True)
    df_1.reset_index()
    df_2.reset_index()
    df_3.reset_index()
    df_4.reset_index()
    TRAIN_SAMPLE = int(len(df_1))
    Y1 = df_1.iloc[:TRAIN_SAMPLE, -1]
    Y2 = df_2.iloc[:TRAIN_SAMPLE, -1]
    Y3 = df_3.iloc[:TRAIN_SAMPLE, -1]
    Y4 = df_4.iloc[:TRAIN_SAMPLE, -1]
    train_dflist_1 = df_1.lyrics[:TRAIN_SAMPLE].tolist()
    train_dflist_2 = df_2.lyrics[:TRAIN_SAMPLE].tolist()
    train_dflist_3 = df_3.lyrics[:TRAIN_SAMPLE].tolist()
    train_dflist_4 = df_4.lyrics[:TRAIN_SAMPLE].tolist()

    lemmatized_words_1 = df_to_lemmatized_words(train_dflist_1)
    lemmatized_words_2 = df_to_lemmatized_words(train_dflist_2)
    lemmatized_words_3 = df_to_lemmatized_words(train_dflist_3)
    lemmatized_words_4 = df_to_lemmatized_words(train_dflist_4)

    #print(lemmatized_words)

    #turn the 2d word list to dict 
    dict_words_1 = words_2d_to_dict(lemmatized_words_1)
    dict_words_2 = words_2d_to_dict(lemmatized_words_2)
    dict_words_3 = words_2d_to_dict(lemmatized_words_3)
    dict_words_4 = words_2d_to_dict(lemmatized_words_4)
    #print(dict_words)

    #there's 2 input for lDA: dictinary and the corpus
    #create bag of words
    corpus_1 =  [dict_words_1.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_1]
    corpus_2 =  [dict_words_2.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_2]
    corpus_3 =  [dict_words_3.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_3]
    corpus_4 =  [dict_words_4.doc2bow(simple_preprocess(" ".join(line))) for line in lemmatized_words_4]

    name = str(every).split('/')[-1].split('.')[0]
    f = open('knn_'+ name +'count.txt', 'a')
    #highest score is gotten when number of topics is 6.
    NUM_TOPICS = 6
    for i in range(2, 20):
        n_neighbors = i
        lda_model_1 = build_LDA_model(dict_words_1, corpus_1)
        lda_model_2 = build_LDA_model(dict_words_2, corpus_2)

        lda_model_3 = build_LDA_model(dict_words_3, corpus_3)

        lda_model_4 = build_LDA_model(dict_words_4, corpus_4)
        print("---------------------start of topic # {}".format(i), "-------------------", file = f)
   
        gen_df_mid_1 = generate_data(lda_model_1, corpus_1)
        gen_df_mid_2 = generate_data(lda_model_2, corpus_2)
        gen_df_mid_3 = generate_data(lda_model_3, corpus_3)
        gen_df_mid_4 = generate_data(lda_model_4, corpus_4)

        #randomForest_LDA(gen_df_mid_1, df_1.iloc[:TRAIN_SAMPLE, -1])

        #kfold for class 1:
        print("KNN classificaton:")
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = kfold_classification(gen_df_mid_1, Y1, knn_clf, zscore_normalization)
    
        #randomForest_LDA(gen_df_mid_2, df_2.iloc[:TRAIN_SAMPLE, -1])
        #kfold for class 2:
  
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = knn_score_sum + kfold_classification(gen_df_mid_2, Y2, knn_clf, zscore_normalization)
    

        #kfold for class 3:
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = knn_score_sum + kfold_classification(gen_df_mid_3, Y3, knn_clf, zscore_normalization)
    
    
        #kfold for class 4:
        rf_clf = RandomForestClassifier()
        knn_clf = KNeighborsClassifier(n_neighbors)
        knn_score_sum = knn_score_sum + kfold_classification(gen_df_mid_4, Y4, knn_clf, zscore_normalization)
    
        knn_score_avg = knn_score_sum / 4
        print("topics: {}, KNN score average of 4 models: {}".format(i, knn_score_avg), file = f)
   
    f.close()


# In[ ]:





# Several parameters to adjust:
# 
# the number of topics;
# 
# alpha and beta: hyperparameter that affect sparsity of the topics. Default to 1.0/num_topics
# 
# chunksize: the number of documents to be used in each training chunk
# 
# update_every: how often the model parameters should be updated
# 
# passes: the total number of training passes

# In[ ]:


# Create the TF-IDF model 
def lda_tfidf_ize_(corpus_list, dict_words_list):
    corpus_tfidf_list = []
    lda_tfidf_model_list = []
    for corpus, dict_words in map(corpus_list, dict_words_list):
        corpus_tfidf_list.append(build_tfidf_model(corpus_1))
        lda_tfidf_model_list.append(dict_words, corpus_tfidf_list[-1])
    return corpus_tfidf_list, lda_tfidf_model_list


# In[ ]:


#visualize the topics

# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model_1, corpus_1, dict_words_1)
# vis


# How to infer pyLDAvis's output?
# 
# Each bubble represents a topic. The larger the bubble, the more prevalent is that topic.
# 
# A good topic model will have fairly big, non-overlapping bubbles scattered throughout the chart instead of being clustered in one quadrant.
# 
# A model with too many topics, will typically have many overlaps, small sized bubbles, clustered in one region of the chart
# 
# So this model ... maybe use fewer number of topics
# 
# You could determine the number of topics using prior knowledge of the number of natural topics in the document

# ## As we can see that after applying tf-idf model as the corpus input, the cohenrence score increased by 0.01, while the classification score decreased by 0.15.

# ## Improve upon this model by using Mallet's version of LDA algorithm, which always gives a better quality of topics than Gensim's inbuilt version of the LDA algorithm.

# In[ ]:


#use mallet's implementation of LDA
def mallet_lda(lemmatized_words, corpus1, dict_words_1, NUM_TOPICS):
    mallet_path = "../mallet-2.0.8/bin/mallet"
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus_1, num_topics=NUM_TOPICS, id2word=dict_words_1)
    # evaluate the mallet model: Compute Coherence Score
    coherence_model = CoherenceModel(model = ldamallet, texts = lemmatized_words_1, dictionary = dict_words_1, coherence = 'c_v')
    coherence_lda = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


# In[ ]:


#try tf-idf on mallet
def mallet_LDA_tfidf(mallet_path, lemmatized_words_1, corpus_tfidf_1, dict_words_1, NUM_TOPICS):
    ldamallet_tfidf = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus_tfidf_1, num_topics=NUM_TOPICS, id2word=dict_words_1)
    # evaluate the mallet model: Compute Coherence Score
    coherence_model = CoherenceModel(model = ldamallet_tfidf, texts = lemmatized_words_1, dictionary = dict_words_1, coherence = 'c_v')
    coherence_lda = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

