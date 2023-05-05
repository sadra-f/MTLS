from statics.paths import *
import pandas as pd
import numpy as np


def print_seperated_file(inp_list, ht_res=None, sb_res=None, cluster_res=None, kmeans_labels=None, tfidf_vector_list=None):
    with open(STR_OUTPUT_PATH, 'w') as opf:
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
        with open(STR_OUTPUT_PATH_DF, 'w') as opf:
            for i in range(len(tfidf_vector_list)):
                tfidf_df = pd.DataFrame(tfidf_vector_list[i][0].toarray(), index=np.where(kmeans_labels == i)[0], 
                columns=tfidf_vector_list[i][1])
                tfidf_df = tfidf_df.sum(axis=0)
                for val in tfidf_df:
                    print(val, file=opf)
                print(file=opf)