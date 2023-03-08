from TimeTagger import HTW
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text as skt
import numpy as np
import pandas as pd
from statics.paths import *
from statics.config import *
from clustering.KMeans_clustering import kmeans, clusterd_inp_sentences, clusterd_inp_list
from IO.Write import print_seperated_file
from Vectorize.sentence_bert import sb_vectorizer as sb






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