
#importing the libraries

import nltk
import numpy as np
import re

import tkinter as tk
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial import distance
#nltk.download('punkt')   # one time execution
#nltk.download('stopwords')  # one time execution
from nltk.corpus import stopwords
#

from nltk.tokenize import sent_tokenize

# cleaning the sentences
def summarizer1(rawdoc):
    text = rawdoc
    sentence = sent_tokenize(text)
    corpus = []
    for i in range(len(sentence)):
        sen = re.sub('[^a-zA-Z]', " ", sentence[i])
        sen = sen.lower()
        sen=sen.split()
        sen = ' '.join([i for i in sen if i not in stopwords.words('english')])
        corpus.append(sen)


    #creating word vectors

    n=300
    all_words = [i.split() for i in corpus]
    model = Word2Vec(all_words, min_count=1)

    # creating sentence vectors

    sen_vector=[]
    for i in corpus:

        plus=0
        for j in i.split():
            plus+=model.wv[j]
        plus = plus/len(plus)

        sen_vector.append(plus)

    #performing k-means

    n_clusters = 2;
    kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(sen_vector)

    #finding and printing the nearest sentence vector from cluster centroid


    my_list=[]
    for i in range(n_clusters):
        my_dict={}

        for j in range(len(y_kmeans)):

            if y_kmeans[j]==i:
                my_dict[j] =  distance.euclidean(kmeans.cluster_centers_[i],sen_vector[j])
        min_distance = min(my_dict.values())
        my_list.append(min(my_dict, key=my_dict.get))


    print(my_list)
    print(y_kmeans)
    doc="";
    j=0;
    for i in sorted(my_list):
        if(j==4):
            break;
        doc+=(sentence[i]);
        j=j+1

    return doc, rawdoc, len(rawdoc.split(' ')), len(doc.split(' ')), len(rawdoc.split('.')), len(doc.split('.'))
