from statics.paths import *
from statics.config import *
from clustering.helpers import cluster_inp_list
from Vector.sentence_bert import sb_vectorizer as sb
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from IO.DocumentRW import DocumentReader
from clustering.DBSCAN import dbscan
from scipy.spatial.distance import euclidean
from TimeTagger.HeidelTime_Generator import ht
import xml.etree.ElementTree as ET
from helpers.distances import *
from helpers.DateParser import DateParser as DP
from helpers.helpers import *

def extract_sentences(doc_list):
    result = []
    for i in range(len(doc_list)):
        doc_ht = ht(doc_list[i].text, date=doc_list[i].date)
        for j in range(len(doc_ht)):
            try:
                xml_tree = ET.fromstring(doc_ht[j])
                if len(xml_tree) > 0 :
                    for tag in xml_tree:
                        doc_list[i].text[j].date = DP.parse(tag.attrib["value"], doc_list[i].date, DO_EXEC_LOG)
                else:
                    doc_list[i].text[j].date = doc_list[i].date
            except Exception as e:
                doc_list[i].text[j].date = doc_list[i].date
            finally:
                result.append(doc_list[i].text[j])
    return result

def main():
    doc_list = DocumentReader(INPUT_PATH, parent_dir_as_date=True).read_all()
    print(len(doc_list))
    sent_list = extract_sentences(doc_list)

    sb_result = sb(sent_list)

    dist = []
    for i in range(len(sent_list)):
        dist.append([])
        sent_list[i].id = i
        sent_list[i].vector = sb_result[i]
        for j in range(len(sent_list)):
            dist[i].append(sentence_distance(sb_result[i], sent_list[i].date, sb_result[j], sent_list[j].date))

    #find smallest/largest distances
    sorted_dist = sort_dist(dist)
    for  i in range(len(sorted_dist)):
        tmp = sorted_dist[i][1:4]
        tmp.append(sorted_dist[i][len(sorted_dist)-4:len(sorted_dist)])
        sorted_dist[i] = tmp
        
    clusters = dbscan(dist, DBSCAN_EPSILON, DBSCAN_MINPOINT)
    
    clustered_sentences = cluster_inp_list(sent_list, clusters.labels, len(clusters.clusters))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    #hold sentence and bert next sentence probability
    bfnsp_cluster_sentence = []

    # test keybert
    cluster_main_phrases = doc_list_keyword_extractor(clustered_sentences)
    
    for i in range(clusters.cluster_count):
        tmp = ""
        for j in range(N_REPRESENTING_PHRASES):
            tmp += f' {cluster_main_phrases[i][j]}'
        cluster_main_phrases[i] = tmp

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
    cluster_vectors = np.zeros((clusters.cluster_count, len(doc_list)))

    for i in range(clusters.cluster_count):
        for j in range(len(clustered_sentences[i])):
            cluster_vectors[i][clustered_sentences[i][j].doc_id] += 1
    
    return
    


if __name__ == '__main__':
    main()