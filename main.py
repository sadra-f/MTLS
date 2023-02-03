from sentence_transformers import SentenceTransformer
from HeidelTime import HTW
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text as skt
import numpy as np


INPUT_PATH = 'IO/input.txt'
OUTPUT_PATH_1 = 'IO/output.txt'
OUTPUT_PATH_2 = 'IO/output_HT.txt'
OUTPUT_PATH_3 = 'IO/output_SB.txt'
OUTPUT_PATH_4 = 'IO/output_km.txt'

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


def main():
    inp = open(INPUT_PATH)
    inp_list = []
    for line in inp:
        inp_list.append(line)
    inp.close()

    sb_res = sb(inp_list)

    KM_model = kmeans(sb_res, N_CLUSTERS)
    
    clustered_sentences = clusterd_inp_list(inp_list, KM_model)
    
    tfidf_per_cluster = []
    for i in range(len(clustered_sentences)):
        #run tfidf on one cluster on each iteration
        tfidf = skt.TfidfVectorizer(input='content', smooth_idf=True, norm=None)#norm='l2' is better
        tfidf_vector = tfidf.fit_transform(clustered_sentences[i])
        tfidf_per_cluster.append(tfidf_vector)


def print_seperated_file(inp_list, ht_res=None, sb_res=None, cluster_res=None):
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


if __name__ == '__main__':
    main()