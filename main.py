from sentence_transformers import SentenceTransformer
from HeidelTime import HTW
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text as skt
import numpy as np
import pandas as pd
from statics.paths import *
from statics.config import *
from clustering.KMeans_clustering import kmeans, clusterd_inp_sentences, clusterd_inp_list
from IO.Write import print_seperated_file



def sb(inp_sentences:list):
    '''
        transforms a list of string into a vector representation with sentenceBert
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(inp_sentences)

    return embeddings


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
    


if __name__ == '__main__':
    main()