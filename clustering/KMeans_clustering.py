from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from statics.config import N_CLUSTERS

def kmeans(vectorized_sentences, n_clusters):
    return KMeans(n_clusters, n_init=80).fit(vectorized_sentences)


def clusterd_inp_list(inp_sentence_list, kmeans_model:KMeans):
    '''
        reformats the initial input list of strings placing strings which are in the 
        same cluster into the same row in a 2d list
    '''
    clustered_sentences = []
    for ci in range(kmeans_model.n_clusters):
        clustered_sentences.append([])
        for inpi in np.where(kmeans_model.labels_ == ci)[0]:
            clustered_sentences[ci].append(inp_sentence_list[inpi])
    
    return clustered_sentences


def cluster_main_words(tfidf_vector_list, km_labels):
    result = []
    for i in range(len(tfidf_vector_list)):
        tmp = pd.DataFrame(tfidf_vector_list[i][0].toarray(), index=np.where(km_labels == i)[0], columns=tfidf_vector_list[i][1])
        tmp = tmp.sum(axis=0).sort_values(ascending=False)
        result.append([])
        #to plot the data accodingly uncomment below
        #tmp.plot(kind='bar')
        for j in range(3):#pick the main 3 words in the list
            result[i].append(tmp.index[j])
    
    return result