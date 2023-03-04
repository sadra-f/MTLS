from sklearn.cluster import KMeans
import numpy as np
from ..statics.config import N_CLUSTERS

def my_kmeans(vectorized_sentences, n_clusters):
    return KMeans(n_clusters, n_init=80).fit(vectorized_sentences)

def clusterd_inp_sentences(inp_sentence_list, kmeans_model:KMeans):
    clustered_inp_list = []
    for ci in range(N_CLUSTERS):
        clustered_inp_list.append([])
        for inpi in np.where(kmeans_model.labels_ == ci)[0]:
            clustered_inp_list[ci].append(inp_sentence_list[inpi])
    return clustered_inp_list


def clusterd_inp_list(inp_sentence_list, kmeans_model:KMeans):
    '''
        reformats the initial input list of strings placing strings in the same cluster into the same row in a 2d list
    '''
    clustered_sentences = []
    for ci in range(kmeans_model.n_clusters):
        clustered_sentences.append([])
        for inpi in np.where(kmeans_model.labels_ == ci)[0]:
            clustered_sentences[ci].append(inp_sentence_list[inpi])
    
    return clustered_sentences