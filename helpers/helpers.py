from keybert import KeyBERT
from statics.stop_words import STOP_WORDS
from statics.config import KEYPHRASE

def sort_dist(dist:list):
    res = []
    for i in range(len(dist)):
        res.append(sorted(dist[i]))
    return res


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