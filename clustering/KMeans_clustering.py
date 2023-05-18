from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from statics.config import *

def kmeans(vectorized_sentences, n_clusters):
    return KMeans(n_clusters, n_init=80).fit(vectorized_sentences)



def cluster_main_words(tfidf_vector_list, labels):
    result = []
    for i in range(len(tfidf_vector_list)):
        tmp = pd.DataFrame(tfidf_vector_list[i][0].toarray(), index=np.where(labels == i)[0], columns=tfidf_vector_list[i][1])
        tmp = tmp.sum(axis=0).sort_values(ascending=False)
        result.append([])
        #to plot the data accodingly uncomment below
        #tmp.plot(kind='bar')
        for j in range(N_REPRESENTING_PHRASES):# 3 is the number of main words in the cluster
            try:
                result[i].append(tmp.index[j])
            except IndexError:
                if j < 1:
                    print('////')
                    print(f'**ERROR** : no main words for cluster (in cluster {i})')
                    print(np.where(labels == i)[0])
                    print('////')
                else:
                    print('////')
                    print(f'**Warning** : Only Found {j+1} main words (in cluster {j})')
                    print(np.where(labels == i)[0])
                    print('////')
                break
    
    return result