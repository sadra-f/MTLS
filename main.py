from statics.paths import *
from statics.config import *
from clustering.KMeans_clustering import kmeans, cluster_inp_list
from IO.Write import print_seperated_file
from Vectorize.sentence_bert import sb_vectorizer as sb
from Vectorize.tfidf import tfidf_list
from transformers import BertTokenizer, BertForNextSentencePrediction
from keybert import KeyBERT
from statics.stop_words import STOP_WORDS
import torch
from IO.Read import DocumentReader
from pathlib import Path
from clustering.testdbscan import dbsacn
from scipy.spatial.distance import euclidean

def keyword_extractor(doc:str):
    model = KeyBERT()
    return model.extract_keywords(doc, keyphrase_ngram_range=(1,2 if KEYPHRASE else 1), stop_words=STOP_WORDS)

def doc_list_keyword_extractor(doc_list:list) -> list[str]:
    res = []
    for cluster_sentence_list in doc_list:
        #tmp
        tmp_str = ""
        for k in cluster_sentence_list:
            tmp_str += k
        #/tmp
        phrase_tuple_list = keyword_extractor(tmp_str)
        res.append([phrase_tuple_list[i][0] for i in range(len(phrase_tuple_list))])
    return res

def main():
    doc_list = DocumentReader(INPUT_PATH, parent_as_date=False).read_all()

    sb_res = sb([doc.text for doc in doc_list])

    KM_model = kmeans(sb_res, N_CLUSTERS)
    
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