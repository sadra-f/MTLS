from statics.paths import *
from statics.config import *
from clustering.KMeans_clustering import kmeans, cluster_inp_list
from IO.Write import print_seperated_file
from Vectorize.sentence_bert import sb_vectorizer as sb
from Vectorize.tfidf import tfidf_list





def main():
    inp = open(INPUT_PATH)
    inp_list = []
    for line in inp:
        inp_list.append(line.lower())
    inp.close()

    sb_res = sb(inp_list)

    KM_model = kmeans(sb_res, N_CLUSTERS)
    
    clustered_sentences = cluster_inp_list(inp_list, KM_model)
    
    cluster_tfidf_vector_list = tfidf_list(clustered_sentences)
    
    print(clustered_sentences)


if __name__ == '__main__':
    main()