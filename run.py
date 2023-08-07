from statics.paths import *
from statics.config import *

from IO.DocumentRW import DocumentReader
from IO.helpers import read_ground_truth
from IO.Read import read_np_array
from IO.Write import write_np_array

from clustering.helpers import cluster_inp_list
from clustering.KMeans_clustering import normal_kmeans, CustomKMeans as KMeans
from clustering.DBSCAN import dbscan
from clustering.NumberClusterFinder import NumberClusterFinder

from helpers.distances import *
from helpers.helpers import *

from models.test import TestClass
from models.ClusteredData import ClusteredData

from Vector.sentence_bert import sb_vectorizer as sb

import torch
import datetime
import numpy as np
import evaluate as eval

from transformers import BertTokenizer, BertForNextSentencePrediction

READ_DIST_FROM_LOG = False
READ_SORTED_DIST_FROM_LOG = True
READ_SB_FROM_LOG = True


def main():
    
    # t1 = np.random.random((27000, 400))
    # t2 = np.ones((27000,1))
    # # t3 = np.ndarray(27000, dtype=tuple)

    
    # dist_metric = distance_metric(metric_type=type_metric.USER_DEFINED, func=test)
    # clstrr = km(t1, [t1[0], t1[1], t1[2]], metric=dist_metric)
    # clstrr.process()
    

    # from sklearn.metrics import pairwise_distances
    # for i in range(len(t1)):
    #     t3[i] = (t1[i], t2[i])
    # 
    # # res[0] = pairwise_distances(t2.reshape(-1,1), metric=test)
    # res = pairwise_distances(t3, t3, metric=test)
    # 
    # print()

    
    doc_list = DocumentReader(DATASET_PATH, parent_dir_as_date=True).read_all()
    DOCUMENT_COUNT = len(doc_list)
    ht_doc_list = DocumentReader(READY_HT_PATH, file_pattern="*.htrs",parent_dir_as_date=True).read_all()
    sent_list = new_extract_sentences(doc_list, ht_doc_list)
    SENTENCE_COUNT = len(sent_list)
    
    if READ_SB_FROM_LOG:
       sb_result = read_np_array(SENTENCE_BERT_VECTORS_PATH)
    else:
        sb_result = sb(sent_list)
    
    # sb_date_tpl_array = np.ndarray(SENTENCE_COUNT, dtype=TestClass)

    if not READ_DIST_FROM_LOG:
        dist = np.zeros((SENTENCE_COUNT, SENTENCE_COUNT))
        for i in range(SENTENCE_COUNT):
            sent_list[i].id = i
            sent_list[i].vector = sb_result[i]
        # write_np_array(dist, CLUSTER1_DIST_PATH)
        initial_sentence_clusters = ClusteredData(KMeans(sent_list, 10, 1).process().labels)
    else:
        dist = read_np_array(CLUSTER1_DIST_PATH)
        for i in range(SENTENCE_COUNT):
            sent_list[i].id = i
            sent_list[i].vector = sb_result[i]    
    

    if READ_SORTED_DIST_FROM_LOG:
        sorted_dist = read_np_array(CLUSTER1_SORTED_DIST_PATH)
    else:
        dist.sort(axis=1)
        sorted_dist = np.flip(dist, axis=1)
    # write_np_array(sorted_dist, CLUSTER1_SORTED_DIST_PATH)
    
    
    for  i in range(len(sorted_dist)):
        tmp = sorted_dist[i][1:4]
        tmp.append(sorted_dist[i][len(sorted_dist)-4:len(sorted_dist)])
        sorted_dist[i] = tmp
     

    eps = NumberClusterFinder(sb_result)
    eps.generateDistance()
    eps.find()
    clusters = dbscan(dist, eps.eps, DBSCAN_MINPOINT_1)
    
     
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
            inputs = tokenizer(clustered_sentences[j][i],cluster_main_phrases[j], return_tensors='pt')
            labels = torch.LongTensor([0])
            outputs = model(**inputs, labels=labels)

            bfnsp_cluster_sentence[j].append((clustered_sentences[j][i], outputs.logits[0][0].item()))
    

    for i in range(clusters.cluster_count):
        bfnsp_cluster_sentence[i] = sorted(bfnsp_cluster_sentence[i], key=lambda x: x[1], reverse=True)
    print(clustered_sentences)

    #build cluster vectors of document percentages
    cluster_vectors = np.zeros((clusters.cluster_count, DOCUMENT_COUNT))
    

    for i in range(clusters.cluster_count):
        for j in range(len(clustered_sentences[i])):
            cluster_vectors[i][clustered_sentences[i][j].doc_id] += 1
      

    cluster_sim = np.zeros((len(cluster_vectors), len(cluster_vectors)))
    for i in range(len(cluster_vectors)) :
        for j in range(len(cluster_vectors)):    
            cluster_sim[i][j] = cluster_distance(cluster_vectors[i], sb_result[sent_list.index(bfnsp_cluster_sentence[i][0][0])], cluster_vectors[j], sb_result[sent_list.index(bfnsp_cluster_sentence[j][0][0])])
    

    eps2 = NumberClusterFinder(cluster_sim)
    eps2.generateDistance()
    eps2.find()
    second_clusters2 = dbscan(cluster_sim, eps2.eps, DBSCAN_MINPOINT_2)
    second_clusters = normal_kmeans(cluster_sim, 2)
    

    gt = [
        [i[1] for i in read_ground_truth("C:\\Users\\TOP\\Desktop\\project\\mtl_dataset\\mtl_dataset\\L2\\D3\\groundtruth\\g1")],
        [i[1] for i in read_ground_truth("C:\\Users\\TOP\\Desktop\\project\\mtl_dataset\\mtl_dataset\\L2\\D3\\groundtruth\\g2")]
    ]
    timelines_clusters = []
    timelines_clusters_sentences = []
    for i in range(second_clusters.cluster_count):
        timelines_clusters.append([])
        timelines_clusters_sentences.append([])
    

    for i in range(len(second_clusters.labels)):
        timelines_clusters[second_clusters.labels[i]].append(i)
        timelines_clusters_sentences[second_clusters.labels[i]].append((bfnsp_cluster_sentence[i][0][0],bfnsp_cluster_sentence[i][0][0].date))
        try:
            timelines_clusters_sentences[second_clusters.labels[i]].append((bfnsp_cluster_sentence[i][1][0],bfnsp_cluster_sentence[i][1][0].date))    
        except:
            continue
    rouge = eval.load('rouge')
    # finals = rouge.compute(predictions=[i[0] for i in timelines_clusters_sentences[1]], references=["demonstrators protest in central cairo", "tunisia also lacked the oil resources of other arab states"])
        
    for i in range(second_clusters.cluster_count):
        for j in range(len(gt)):
            prd = [i[0] for i in timelines_clusters_sentences[i]]
            size = len(prd) if len(prd) < len(gt[j]) else len(gt[j])
            metrics11 = rouge.compute(predictions=prd[:size], references=gt[j][:size])
            metrics12 = rouge.compute(predictions=[prd], references=[gt[j]])
            print(f'generated({i}) GT({j}) ==> ', metrics11)
            print(f'generated({i}) GT({j}) ==> ', metrics12)
            try:
                metrics21 = rouge.compute(predictions=prd[:size], references=gt[j][:size])
                metrics22 = rouge.compute(predictions=[prd], references=[gt[j]])
                print(f'generated({i}) GT({j}) ==> ', metrics21)
                print(f'generated({i}) GT({j}) ==> ', metrics22)
            except:
                continue
    return
    


if __name__ == '__main__':
    main()