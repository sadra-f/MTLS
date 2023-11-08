from statics.paths import *
from statics.config import *

from IO.DocumentRW import DocumentReader
from IO.Read import read_np_array, read_all_GTs
from IO.Write import write_np_array

from clustering.helpers import cluster_inp_list, dbscan_eps
from clustering.KMeans_clustering import normal_kmeans, CustomKMeans as KMeans, CustomKMeans2 as KMeans2
from clustering.DBSCAN import dbscan

from helpers.distances import *
from helpers.helpers import *

from models.ClusteredData import ClusteredData
from models.ClusterDistance import DistanceKmeans
from Vector.sentence_bert import sb_vectorizer as sb

import torch
import datetime
import numpy as np
import evaluate as eval
from itertools import combinations
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 
from transformers import BertTokenizer, BertForNextSentencePrediction
from models.ClusterDistance import DistanceKmeans

READ_DIST_FROM_LOG = False
READ_SB_FROM_LOG = False

def main():
    print('init : ', datetime.datetime.now())
    doc_list = DocumentReader(DATASET_PATH, parent_dir_as_date=True).read_all()
    DOCUMENT_COUNT = len(doc_list)
    ht_doc_list = DocumentReader(READY_HT_PATH, file_pattern="*.htrs",parent_dir_as_date=True).read_all()
    sent_list = new_extract_sentences(doc_list, ht_doc_list)
    if READ_SB_FROM_LOG:
        sb_result = read_np_array(SENTENCE_BERT_VECTORS_PATH)
    else:
        sb_result = sb(sent_list)
    
    for i, bert in enumerate(sb_result):
        sent_list[i].id = i
        sent_list[i].vector = bert

    if not READ_DIST_FROM_LOG:
        init_KM_clusters = ClusteredData(KMeans(sent_list, 5 * N_TIMELINES, 3).process().labels)
        print('kmeans: ', datetime.datetime.now())
        # pt = PCA(2)
        # t = pt.fit(sb_result)
        # for i, val in enumerate(initial_sentence_clusters.seperated):
        #     plt.scatter([t[i][0] for i in val],[t[i][1] for i in val],label=i)
        init_clustered_sentences = cluster_inp_list(sent_list, init_KM_clusters.labels, init_KM_clusters.cluster_count)
        
        dists = np.full((init_KM_clusters.cluster_count,), None, dtype=object)
        for i, cluster in enumerate(init_clustered_sentences):
            dists[i] = np.zeros((len(cluster),len(cluster)), dtype=np.float16)
            for j in range(len(cluster)):
                for k in range(j, len(cluster), 1):
                    dists[i][j][k] = dists[i][k][j] = sentence_distance(cluster[j], cluster[k])
            print('distances: ', datetime.datetime.now())

    else:
        dists = read_np_array(CLUSTER1_DIST_PATH)
        init_KM_clusters = ClusteredData(read_np_array(INIT_CLUSTER_LABELS_PATH))
        init_clustered_sentences = read_np_array(INIT_CLUSTER_SENT_PATH)
    

    clustered_sentences = []
    for i in range(init_KM_clusters.cluster_count):
        eps = dbscan_eps(dists[i], DBSCAN_MINPOINT_1)
        clusters = dbscan(dists[i], eps, DBSCAN_MINPOINT_1)        
        clustered_sentences.extend(cluster_inp_list(init_clustered_sentences[i], clusters.labels, clusters.cluster_count))


    #hold sentence and bert next sentence probability
    bfnsp_cluster_sentence = []
    
    # keybert key phrase finder
    cluster_main_phrases = doc_list_keyword_extractor(clustered_sentences)
    
    for i in range(clusters.cluster_count):
        tmp = ""
        for j in range(N_REPRESENTING_PHRASES):
            try:
                tmp += f' {cluster_main_phrases[i][j]}'
            except IndexError as e:
                tmp += ''
        cluster_main_phrases[i] = tmp
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    for j in range(clusters.cluster_count):
        bfnsp_cluster_sentence.append([])
        for i in range(len(clustered_sentences[j])):
            inputs = tokenizer(clustered_sentences[j][i], cluster_main_phrases[j], return_tensors='pt')
            labels = torch.LongTensor([0])
            outputs = model(**inputs, labels=labels)

            bfnsp_cluster_sentence[j].append((clustered_sentences[j][i], outputs.logits[0][0].item()))
    

    for i in range(clusters.cluster_count):
        bfnsp_cluster_sentence[i] = sorted(bfnsp_cluster_sentence[i], key=lambda x: x[1], reverse=True)

    #build cluster vectors of document percentages
    cluster_vectors = np.zeros((clusters.cluster_count, DOCUMENT_COUNT))
    

    for i in range(clusters.cluster_count):
        for j in range(len(clustered_sentences[i])):
            cluster_vectors[i][clustered_sentences[i][j].doc_id] += 1
      

    cluster_sim = np.zeros((len(cluster_vectors), len(cluster_vectors)))
    for i in range(len(cluster_vectors)) :
        for j in range(len(cluster_vectors)):    
            cluster_sim[i][j] = cluster_distance(cluster_vectors[i], bfnsp_cluster_sentence[i][0][0].vector, cluster_vectors[j], bfnsp_cluster_sentence[j][0][0].vector)
    clusternig_input = []
    for i in range(len(cluster_vectors)):
        clusternig_input.append(DistanceKmeans(cluster_vectors[i], bfnsp_cluster_sentence[i][0][0]))

    # second_clusters = normal_kmeans(cluster_sim, 2)
    # eps2 = dbscan_eps(cluster_sim, DBSCAN_MINPOINT_2)
    # second_clusters = dbscan(cluster_sim, eps2, DBSCAN_MINPOINT_2)
    second_clusters = ClusteredData(KMeans2(clusternig_input, N_TIMELINES, 5).process().labels)

    gt = read_all_GTs(DATASET_PATH, N_TIMELINES)
    
    timelines_clusters_sentences = []
    for i in range(second_clusters.cluster_count):
        timelines_clusters_sentences.append([])
    

    for i in range(len(second_clusters.labels)):
        timelines_clusters_sentences[second_clusters.labels[i]].append((bfnsp_cluster_sentence[i][0][0],bfnsp_cluster_sentence[i][0][0].date))
        try:
            timelines_clusters_sentences[second_clusters.labels[i]].append((bfnsp_cluster_sentence[i][1][0],bfnsp_cluster_sentence[i][1][0].date))    
        except:
            continue

    rouge = eval.load('rouge')
    evaluations = np.ndarray((second_clusters.cluster_count, len(gt)), dtype=object)
    for i in range(second_clusters.cluster_count):
        for j in range(len(gt)):
            prd = [k[0] for k in timelines_clusters_sentences[i]]
            evaluation = rouge.compute(predictions=[' '.join(prd)], references=[' '.join(gt[j])])
            evaluations[i][j] = evaluation

    print(evaluations)
    print(datetime.datetime.now())
    return
    


if __name__ == '__main__':
    main()