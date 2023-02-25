from sentence_transformers import SentenceTransformer
from HeidelTime import HTW
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text as skt
import numpy as np
import pandas as pd
from statics.paths import *

N_CLUSTERS = 5

def ht(input, reformat:bool=True):
    '''
        transforms a text or a list of text into a time tagged version of it
    '''
    hw = HTW('english', reformat_output=reformat)
    res_list = []
    if type(input) == list:
        for string in input:
            res_list.append(hw.parse(string))
        return res_list
    elif type(input) == str:
        return hw.parse(input)
    else:
        raise TypeError


def sb(inp_sentences:list):
    '''
        transforms a list of string into a vector representation with sentenceBert
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(inp_sentences)

    return embeddings


def kmeans(vectorized_sentences, n_clusters):
    return KMeans(n_clusters, n_init=80).fit(vectorized_sentences)

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

def clusterd_inp_sentences(inp_sentence_list, kmeans_model:KMeans):
    clustered_inp_list = []
    for ci in range(kmeans_model.n_clusters):
        clustered_inp_list.append([])
        for inpi in np.where(kmeans_model.labels_ == ci)[0]:
            clustered_inp_list[ci].append(inp_sentence_list[inpi])
    return clustered_inp_list

def tfidf(doc):
    # runs tfdif on a list of strings (a document)
    tfidf = skt.TfidfVectorizer(input='content', smooth_idf=True, norm='l2')
    return (tfidf.fit_transform(doc), tfidf.get_feature_names_out())

def tfidf_list(doc_list):
    #runs tfidf on a list of list of strings (multiple documents)
    res_list = []
    for i in range(len(doc_list)):
        res_list.append(tfidf(doc_list[i]))

    return res_list



def cluster_main_words(tfidf_vector_list, km_labels):
    result = []
    for i in range(len(tfidf_vector_list)):
        tmp = pd.DataFrame(tfidf_vector_list[i][0].toarray(), index=np.where(km_labels == i)[0], columns=tfidf_vector_list[i][1])
        tmp = tmp.sum(axis=0).sort_values(ascending=False)
        result.append([])
        #to plot the data accodingly
        #tmp.plot(kind='bar')
        for j in range(3):#pick the main 3 words in the list
            result[i].append(tmp.index[j])
    
    return result

def main():
    inp = open(INPUT_PATH)
    inp_list = []
    for line in inp:
        inp_list.append(line.lower())
    inp.close()

    sb_res = sb(inp_list)

    KM_model = kmeans(sb_res, N_CLUSTERS)
    
    clustered_sentences = clusterd_inp_list(inp_list, KM_model)
    
    clustered_sentences = clusterd_inp_sentences(inp_list, KM_model)
    
    cluster_tfidf_vector_list = tfidf_list(clustered_sentences)
    
def print_seperated_file(inp_list, ht_res=None, sb_res=None, cluster_res=None, kmeans_labels=None, tfidf_vector_list=None):
    with open(OUTPUT_PATH_1, 'w') as opf:
        for i in range(len(inp_list)):
            print("### ORIGINAL", file=opf)
            print(inp_list[i], file=opf)
            if ht_res is not None:
                print("### HeidelTime", file=opf)
                print(ht_res[i], file=opf)
            if sb_res is not None:
                print("### SentenceBert", file=opf)
                print('[ ', file=opf, end='')
                for j in range(len(sb_res[i])):
                    print(sb_res[i][j], file=opf, end=' ')
                print(']', file=opf)
            if cluster_res is not None:
                print(f'### Cluster Number ==> #{cluster_res[i]}', file=opf)
            print('=================================================================', file=opf)
    if tfidf_vector_list is not None and kmeans_labels is not None:
        with open(OUTPUT_PATH_5, 'w') as opf:
            for i in range(len(tfidf_vector_list)):
                tfidf_df = pd.DataFrame(tfidf_vector_list[i][0].toarray(), index=np.where(kmeans_labels == i)[0], 
                columns=tfidf_vector_list[i][1])
                print(tfidf_df.style.render(), file=opf)

if __name__ == '__main__':
    main()