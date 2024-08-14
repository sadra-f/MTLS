from statics.paths import *
from statics.config import *

from IO.DocumentRW import DocumentReader
from IO.Read import read_np_array, read_all_GTs
from IO.Write import write_np_array
#........
# from IO.helpers import print_2d_array as p2a, print_1d_array as p1a
from helpers.helpers import main_phrase_counter as mpc, clust_subj_vec as clsv, cancat_by_date as cbd
from nltk.stem import SnowballStemmer
#.......
from clustering.helpers import cluster_inp_list, dbscan_eps
from clustering.KMeans_clustering import normal_kmeans, CustomKMeans as KMeans, AltCustomKMeans as AKMeans
from clustering.DBSCAN import dbscan

from helpers.distances import *
from helpers.helpers import *

from models.ClusteredData import ClusteredData
from models.DXCV import DocumentxClusterVector as DXCV
from Vector.sentence_bert import sb_vectorizer as sb

import torch
import datetime
import numpy as np
from transformers import BertTokenizer, BertForNextSentencePrediction

from Evaluation.Concat import concat_rouge
from Evaluation.Align_m1 import align_m1_rouge
from Evaluation.DSelect import calculate_d_select
from itertools import chain

READ_SB_FROM_LOG = False
READ_DIST_FROM_LOG = False
READ_BFNSP_FROM_LOG = False

# READ_TIMELINES = False

DO_SAVE_RESULTS = True

def main():
    print('init : ', datetime.datetime.now())

    doc_list = DocumentReader(DATASET_PATH, parent_dir_as_date=True).read_all()
    DOCUMENT_COUNT = len(doc_list)

    ht_doc_list = DocumentReader(READY_HT_PATH, file_pattern="*.htrs",parent_dir_as_date=True).read_all()

    sent_list = new_extract_sentences(doc_list, ht_doc_list)
    gt = read_all_GTs(DATASET_PATH, N_TIMELINES)

    if READ_SB_FROM_LOG:
        sb_result = read_np_array(SENTENCE_BERT_VECTORS_PATH, N_TIMELINES, DATASET_NUMBER)
        gt_sb_result = read_np_array(GT_SENTENCE_BERT_VECTORS_PATH, N_TIMELINES, DATASET_NUMBER)
    else:
        non_gt_count = len(sent_list)
        all_sents = sent_list + list(chain.from_iterable(gt))
        all_sb_res = sb(all_sents)
        sb_result = all_sb_res[:non_gt_count]
        gt_sb_result = all_sb_res[non_gt_count:]
        write_np_array(sb_result, SENTENCE_BERT_VECTORS_PATH, N_TIMELINES, DATASET_NUMBER)
        write_np_array(gt_sb_result, GT_SENTENCE_BERT_VECTORS_PATH, N_TIMELINES, DATASET_NUMBER)
    
    for i, bert in enumerate(sb_result):
        sent_list[i].id = i
        sent_list[i].vector = bert
    cntr = 0
    for i, clus in enumerate(gt):
        for j, str in enumerate(clus):
            gt[i][j].vector = gt_sb_result[cntr]
            cntr +=1

    # remove duplicate strings and reorder
    sent_list = sorted(list(set(sent_list)), key= lambda x: x.id)
    
    if not READ_DIST_FROM_LOG:
        init_KM_clusters = ClusteredData(KMeans(sent_list, INITIAL_KMEANS_TIMLINE_MULTIPLIER * N_TIMELINES, 5).process().labels)
        print('kmeans: ', datetime.datetime.now())

        init_clustered_sentences = cluster_inp_list(sent_list, init_KM_clusters.labels, init_KM_clusters.cluster_count)


        dists = np.full((init_KM_clusters.cluster_count,), None, dtype=object)
        for i, cluster in enumerate(init_clustered_sentences):
            dists[i] = np.zeros((len(cluster),len(cluster)), dtype=np.float16)
            for j in range(len(cluster)):
                for k in range(j, len(cluster), 1):
                    dists[i][j][k] = dists[i][k][j] = sentence_distance(cluster[j], cluster[k])
            print('distances: ', datetime.datetime.now())
        
        write_np_array(dists, CLUSTER1_DIST_PATH, N_TIMELINES, DATASET_NUMBER)
        write_np_array(init_KM_clusters.labels, INIT_CLUSTER_LABELS_PATH, N_TIMELINES, DATASET_NUMBER)
        write_np_array(init_clustered_sentences, INIT_CLUSTER_SENT_PATH, N_TIMELINES, DATASET_NUMBER)

    else:
        dists = read_np_array(CLUSTER1_DIST_PATH, N_TIMELINES, DATASET_NUMBER)
        init_KM_clusters = ClusteredData(read_np_array(INIT_CLUSTER_LABELS_PATH, N_TIMELINES, DATASET_NUMBER))
        init_clustered_sentences = read_np_array(INIT_CLUSTER_SENT_PATH, N_TIMELINES, DATASET_NUMBER)
    

    clustered_sentences = []
    for i in range(init_KM_clusters.cluster_count):
        eps = dbscan_eps(dists[i], DBSCAN_MINPOINT_1)
        clusters = dbscan(dists[i], eps, DBSCAN_MINPOINT_1)        
        clustered_sentences.extend(cluster_inp_list(init_clustered_sentences[i], clusters.labels, clusters.cluster_count))
    
    FIRST_CLUSTER_COUNT = len(clustered_sentences)
    print("Event Count : ", FIRST_CLUSTER_COUNT)

    #hold sentence and bert next sentence probability
    bfnsp_cluster_sentence = []
    cluster_main_phrases = None
    subjs = None
    if not READ_BFNSP_FROM_LOG:
        # keybert key phrase finder
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        
        cluster_main_phrases = doc_list_kewords_sentence(clustered_sentences)
        subjs = mpc(cluster_main_phrases)

        cleaned_events = []
        counts = []

        cleanup_value = CLEANUP_PERCENTAGE * (FIRST_CLUSTER_COUNT / 100)
        cleanup_value = 1 if cleanup_value < 1 else cleanup_value
        cleanup_value = CLEANUP_CONST if cleanup_value > CLEANUP_CONST else cleanup_value
        for i, event in enumerate(clustered_sentences):
            count = 0
            for subj in subjs:
                count += len(re.findall(f"{subj[1]}", " ".join(event)))
            counts.append(count)
            if count >= cleanup_value:
                cleaned_events.append(event)
        clustered_sentences = cleaned_events
        FIRST_CLUSTER_COUNT = len(clustered_sentences)

        for j in range(FIRST_CLUSTER_COUNT):
            bfnsp_cluster_sentence.append([])
            for i in range(len(clustered_sentences[j])):
                try:    
                    inputs = tokenizer(clustered_sentences[j][i], cluster_main_phrases[j], return_tensors='pt')
                    labels = torch.LongTensor([0])
                    outputs = model(**inputs, labels=labels)
                except:
                    bfnsp_cluster_sentence[j].append((clustered_sentences[j][i], -10))

                bfnsp_cluster_sentence[j].append((clustered_sentences[j][i], outputs.logits[0][0].item()))

            bfnsp_cluster_sentence[j] = sorted(bfnsp_cluster_sentence[j], key=lambda x: x[1], reverse=True)
        
        write_np_array(bfnsp_cluster_sentence, BFNSP_RES_PATH, N_TIMELINES, DATASET_NUMBER)
        write_np_array(clustered_sentences, CLUSTER1_CLEAN_SENTENCES_PATH, N_TIMELINES, DATASET_NUMBER)
    else:
        bfnsp_cluster_sentence = read_np_array(BFNSP_RES_PATH, N_TIMELINES, DATASET_NUMBER).tolist()
        clustered_sentences = read_np_array(CLUSTER1_CLEAN_SENTENCES_PATH, N_TIMELINES, DATASET_NUMBER).tolist()

    FIRST_CLUSTER_COUNT = len(bfnsp_cluster_sentence)
    #build cluster vectors of document percentages
    cluster_vectors = np.full((FIRST_CLUSTER_COUNT, ), None, object)

    for i in range(FIRST_CLUSTER_COUNT):
        cluster_vectors[i] = DXCV(np.zeros((DOCUMENT_COUNT,)), bfnsp_cluster_sentence[i][0][0].vector)
        for j in range(len(clustered_sentences[i])):
            cluster_vectors[i].doc_cluster_vector[clustered_sentences[i][j].doc_id] += 1


    second_clusters = ClusteredData(AKMeans(cluster_vectors, N_TIMELINES, 5).process().labels)

    timelines_clusters_sentences = [ [] for i in range(second_clusters.cluster_count)]

    for i in range(len(second_clusters.labels)):
        timelines_clusters_sentences[second_clusters.labels[i]].append(bfnsp_cluster_sentence[i][0][0])

    evaluations_concat = concat_rouge(timelines_clusters_sentences, gt)
    evaluations_alignm1 = align_m1_rouge(cbd(timelines_clusters_sentences), cbd(gt))
    evaluations_d_select = calculate_d_select(timelines_clusters_sentences, gt)

    print(evaluations_concat)
    print(evaluations_alignm1)
    print(evaluations_d_select)
    print(datetime.datetime.now())
    if DO_SAVE_RESULTS:
        with open(f'../Results/concat/L{N_TIMELINES}D{DATASET_NUMBER}.txt', 'w') as f:
            print(f'{evaluations_concat}', file=f)

        with open(f'../Results/alignm1/L{N_TIMELINES}D{DATASET_NUMBER}.txt', 'w') as f:
            print(f'{evaluations_alignm1}', file=f)

        with open(f'../Results/dselect/L{N_TIMELINES}D{DATASET_NUMBER}.txt', 'w') as f:
            print(f'{evaluations_d_select}', file=f)

        with open(f'../Result_timelines/L{N_TIMELINES}D{DATASET_NUMBER}.txt', 'w') as f:
            for i in timelines_clusters_sentences:
                for j in i:
                    print(f'{j[0]} ==> ({j[1]})', file=f)
                print("\r\n", file=f)
    return
    


if __name__ == '__main__':
    main()    