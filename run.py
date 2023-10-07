from statics.paths import *
from statics.config import *

from IO.DocumentRW import DocumentReader
from IO.helpers import read_ground_truth
from IO.Read import read_np_array
from IO.Write import write_np_array

from clustering.helpers import cluster_inp_list, dbscan_eps
from clustering.KMeans_clustering import normal_kmeans, CustomKMeans as KMeans
from clustering.DBSCAN import dbscan

from helpers.distances import *
from helpers.helpers import *

from models.ClusteredData import ClusteredData

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

READ_DIST_FROM_LOG = True
# READ_SORTED_DIST_FROM_LOG = False
READ_SB_FROM_LOG = True


def main():
    print(datetime.datetime.now())
    doc_list = DocumentReader(DATASET_PATH, parent_dir_as_date=True).read_all()
    DOCUMENT_COUNT = len(doc_list)
    ht_doc_list = DocumentReader(READY_HT_PATH, file_pattern="*.htrs",parent_dir_as_date=True).read_all()
    sent_list = new_extract_sentences(doc_list, ht_doc_list)
    SENTENCE_COUNT = len(sent_list)
    if READ_SB_FROM_LOG:
        sb_result = read_np_array(SENTENCE_BERT_VECTORS_PATH)
    else:
        sb_result = sb(sent_list)
    
    for i, bert in enumerate(sb_result):
        sent_list[i].id = i
        sent_list[i].vector = bert

    if not READ_DIST_FROM_LOG:
        dist = np.full((SENTENCE_COUNT, SENTENCE_COUNT), np.finfo(np.float64).max)
        print(datetime.datetime.now())
        initial_sentence_clusters = ClusteredData(KMeans(sent_list, 10, 3).process().labels)
        # pt = PCA(2)
        # t = pt.fit(sb_result)
        # for i, val in enumerate(initial_sentence_clusters.seperated):
        #     plt.scatter([t[i][0] for i in val],[t[i][1] for i in val],label=i)
        for cluster in initial_sentence_clusters.seperated:
            print(datetime.datetime.now())
            for j, k in combinations(cluster, 2):
                    dist[j][k] = dist[k][j] = sentence_distance(sent_list[j], sent_list[k])
        # write_np_array(dist, CLUSTER1_DIST_PATH)

    else:
        dist = read_np_array(CLUSTER1_DIST_PATH)
    

    for i, vec in enumerate(dist):
        vec[vec == np.inf] = 100
        vec[vec < 0] = 0
        vec[i] = 0
        
    eps = dbscan_eps(dist, DBSCAN_MINPOINT_1)

    clusters = dbscan(dist, eps, DBSCAN_MINPOINT_1)
     
    clustered_sentences = cluster_inp_list(sent_list, clusters.labels, clusters.cluster_count)


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
            cluster_sim[i][j] = cluster_distance(cluster_vectors[i], sb_result[sent_list.index(bfnsp_cluster_sentence[i][0][0])], cluster_vectors[j], sb_result[sent_list.index(bfnsp_cluster_sentence[j][0][0])])
    
    # cluster2=[]
    # for i in range(len(cluster_vectors)):
    #     clusterkmeans=DistanceKmeans(cluster_vectors[i],sb_result[sent_list.index(bfnsp_cluster_sentence[i][0][0])])
    #     cluster2.append(clusterkmeans)
     
    eps2 = dbscan_eps(cluster_sim, DBSCAN_MINPOINT_2)
    second_clusters = dbscan(cluster_sim, eps2, DBSCAN_MINPOINT_2)
    # second_sentence_clusters = ClusteredData(CustomKMeans2(cluster2, 3, 5).process().labels)

    gt = [
        [i[1] for i in read_ground_truth("C:\\Users\\TOP\\Desktop\\project\\mtl_dataset\\mtl_dataset\\L2\\D3\\groundtruth\\g1")],
        [i[1] for i in read_ground_truth("C:\\Users\\TOP\\Desktop\\project\\mtl_dataset\\mtl_dataset\\L2\\D3\\groundtruth\\g2")]
    ]
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
    # finals = rouge.compute(predictions=[i[0] for i in timelines_clusters_sentences[1]], references=["demonstrators protest in central cairo", "tunisia also lacked the oil resources of other arab states"])
    evaluations = np.ndarray((second_clusters.cluster_count, len(gt)), dtype=object)
    for i in range(second_clusters.cluster_count):
        for j in range(len(gt)):
            prd = [k[0] for k in timelines_clusters_sentences[k]]
            size = len(prd) if len(prd) < len(gt[j]) else len(gt[j])
            evaluation = rouge.compute(predictions=prd[:size], references=gt[j][:size])
            evaluations[i][j] = evaluation
            # metrics12 = rouge.compute(predictions=[prd], references=[gt[j]])

    print(evaluations)
    print(datetime.datetime.now())
    return
    


if __name__ == '__main__':
    main()