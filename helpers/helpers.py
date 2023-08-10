from statics.stop_words import STOP_WORDS
from statics.config import KEYPHRASE
from statics.config import DO_EXEC_LOG

from helpers.DateParser import DateParser as DP

from TimeTagger.HeidelTime_Generator import ht

import xml.etree.ElementTree as ET
from keybert import KeyBERT
import re
import numpy as np

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

def half_matrix_to_full(matrix, dims, diagonal):
    for i in range(dims):
        j = i + 1
        matrix[i][j] = diagonal
        while j < dims:
            matrix[i][j] = matrix[j][i]
            j+= 1

def check_samepath(filename1, filename2):
    indx1 = re.search("\\\\L\d{1}\\\\", str(filename1))
    indx2 = re.search("\\\\L\d{1}\\\\", str(filename2))
    if filename1[indx1.span()[0]:] == filename2[indx2.span()[0]:]:
        return True
    return False



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


def new_extract_sentences(doc_list, HT_list):
    result = []
    for i in range(len(doc_list)):
        for j in range(len(HT_list[i].text)):
            try:
                # for k in re.finditer("<[^<]+>( |$|)", HT_list[i].text[j]):
                #     HT_list[i].text[j] = (HT_list[i].text[j])[:k.span()[1]-1] + '\n' + (HT_list[i].text[j])[k.span()[1]:]
                
                HT_list[i].text[j] = HT_list[i].text[j].replace('!doctype timeml system "timeml.dtd"', '!DOCTYPE TimeML SYSTEM "TimeML.dtd"')
                HT_list[i].text[j] = HT_list[i].text[j].replace('timeml', 'TimeML')
                HT_list[i].text[j] = HT_list[i].text[j].replace('timex3', 'TIMEX3')
                HT_list[i].text[j] = HT_list[i].text[j].replace("\'", "")
                HT_list[i].text[j] = HT_list[i].text[j].replace("&", "and")
                xml_tree = ET.fromstring(HT_list[i].text[j])
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


def array_compressor(array):
    res = [len(array)]
    counter = 0
    for i, value in enumerate(array):
        if value == np.inf:
            counter += 1
            continue
        elif counter > 2:
            res.append(np.NAN)
            res.append(counter)
            counter = 0
        else:
            res.append(np.inf)
            res.append(np.inf)
        res.append(value)
    return np.array(res, dtype=np.float64)