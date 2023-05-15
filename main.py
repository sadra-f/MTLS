from statics.paths import *
from statics.config import *
from clustering.KMeans_clustering import kmeans, cluster_inp_list
from IO.Write import print_seperated_file
from Vectorize.sentence_bert import sb_vectorizer as sb
from Vectorize.tfidf import tfidf_list
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from IO.Read import DocumentReader
from pathlib import Path
from clustering.testdbscan import dbsacn
from scipy.spatial.distance import euclidean
from TimeTagger.HeidelTime_Generator import ht
from datetime import date, datetime, timedelta
import xml.etree.ElementTree as ET
from numpy import dot, ndarray
from numpy.linalg import norm
import numpy as np
from distances import sentence_distance
from DateParser import DateParser as DP
from helpers import *


def main():
    doc_list = DocumentReader(INPUT_PATH, parent_dir_as_date=False).read_all()
    print(len(doc_list))
    sent_list = []
    for i in range(len(doc_list)):
        doc_ht = ht(doc_list[i].text, date=doc_list[i].date)
        for j in range(len(doc_ht)):
            try:
                xml_tree = ET.fromstring(doc_ht[j])
                if len(xml_tree) > 0 :
                    for tag in xml_tree:
                        doc_list[i].text[j].date = DP.parse(tag.attrib["value"], doc_list[i].date, DO_LOG)
                else:
                    doc_list[i].text[j].date = doc_list[i].date
            except Exception as e:
                doc_list[i].text[j].date = doc_list[i].date
            finally:
                sent_list.append(doc_list[i].text[j])

    sb_result = sb(sent_list)

    dist = []
    for i in range(len(sent_list)):
        dist.append([])
        for j in range(len(sent_list)):
            dist[i].append(sentence_distance(sb_result[i], sent_list[i].date, sb_result[j], sent_list[j].date))

    strd = sort_dist(dist)
    for  i in range(len(strd)):
        tmp = strd[i][1:4]
        tmp.append(strd[i][len(strd)-4:len(strd)])
        strd[i] = tmp
        
    TMP = dbsacn(dist, DBSCAN_EPSILON, DBSCAN_MINPOINT)
    print(dbsacn(dist, DBSCAN_EPSILON, DBSCAN_MINPOINT))

    # KM_model = kmeans(sb_res[0], N_CLUSTERS)
    
    clustered_sentences = cluster_inp_list([doc.text for doc in doc_list], KM_model)
    
    cluster_tfidf_vector_list = tfidf_list(clustered_sentences)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    #hold sentence and bert next sentence probability
    bfnsp_cluster_sentence = []

    # test keybert
    cluster_main_phrases = doc_list_keyword_extractor(clustered_sentences)

    for i in range(N_CLUSTERS):
        tmp = ""
        for j in range(N_REPRESENTING_PHRASES):
            tmp += f' {cluster_main_phrases[i][j]}'
        cluster_main_phrases[i] = tmp

        for j in range(N_CLUSTERS):
            bfnsp_cluster_sentence.append([])
            for i in range(len(clustered_sentences[j])):
                inputs = tokenizer(clustered_sentences[j][i],cluster_main_phrases[j], return_tensors='pt')
                labels = torch.LongTensor([0])
                outputs = model(**inputs, labels=labels)

                bfnsp_cluster_sentence[j].append((clustered_sentences[j][i], outputs.logits[0][0].item()))

    for i in range(N_CLUSTERS):
        bfnsp_cluster_sentence[i] = sorted(bfnsp_cluster_sentence[i], key=lambda x: x[1], reverse=True)
    print(clustered_sentences)

    for i in range(N_CLUSTERS):
        for j in range(3):#3 is the number of representing sentences in cluster
            print()
            print(bfnsp_cluster_sentence[i][j])
            print()
        print("#####################################")
    


if __name__ == '__main__':
    main()