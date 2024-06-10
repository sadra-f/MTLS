from statics.stop_words import STOP_WORDS
from statics.config import KEYPHRASE, N_REPRESENTING_PHRASES
from statics.config import DO_EXEC_LOG

from helpers.DateParser import DateParser as DP
#..........
from nltk.stem import SnowballStemmer as SS
import datetime
#..........
from TimeTagger.HeidelTime_Generator import ht

import xml.etree.ElementTree as ET
from keybert import KeyBERT
import re
import numpy as np

def sort_dist(dist:list):
    """
        sorts a list of distance values
        previously used
    """
    res = []
    for i in range(len(dist)):
        res.append(sorted(dist[i]))
    return res


def keyword_extractor(doc:str):
    """
        using a bert model (KeyBert) extracts key words/phrases form a document
    """
    model = KeyBERT()
    return model.extract_keywords(doc, keyphrase_ngram_range=(1,2 if KEYPHRASE else 1), stop_words=STOP_WORDS, top_n=N_REPRESENTING_PHRASES)


def doc_list_keyword_extractor(doc_list:list) -> list[str]:
    """extracts the keywords for the docuemnts/sentences in each of thec clusters/lists in the given list

    Args:
        doc_list (list): the list of clusters list(list(sentence))

    Returns:
        list[str]: the key words for each cluster/list list(list(keyword))
    """
    return keyword_extractor([' '.join(docs) for docs in doc_list])

def doc_list_kewords_sentence(doc_list:list) -> list[str]:
    keywords = doc_list_keyword_extractor(doc_list)
    for i in range(len(keywords)):
        keywords[i] = ' '.join([keyword[0] for keyword in keywords[i]])
    
    return keywords


def half_matrix_to_full(matrix, dims, diagonal):
    """Not USED
            fills a matrix with the values in one diagonal half

            example: 
                this:
                [0, 0, 0, 0]
                [1, 0, 0, 0]
                [2, 4, 0, 0]
                [3, 5, 6, 0]
                
                trurns into:
                [0, 1, 2, 3]
                [1, 0, 4, 5]
                [2, 4, 0, 6]
                [3, 5, 6, 0]


    Args:
        matrix (_type_): the matrix to fill
        dims (_type_): the value for the first dim
        diagonal (_type_): value for the diagonal line
    """
    for i in range(dims):
        j = i + 1
        matrix[i][j] = diagonal
        while j < dims:
            matrix[i][j] = matrix[j][i]
            j+= 1

def check_samepath(filename1, filename2):
    """checks if two paths are the same

    Args:
        filename1 (_type_): first path to compare
        filename2 (_type_): second path to compare

    Returns:
        _type_: _description_
    """
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
    """extracts the sentences and their dates from precomputed heideltime timeml outputs

    Args:
        doc_list (list): list of the documetns
        HT_list (list): list of the ht results matching sentences element-wise

    Returns:
        _type_: list of docuemnts with their dates fixed to them
    """
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

class Compress:

    def vector_compressor(vector, compress=np.inf):
        """
            read through the vector and convert multiple consecutive occurrences of np.inf into counters of it
            example:
                inf, inf, inf, inf, inf, 0.2, 0,3, inf, 0.5
            will be:
                nan, 5, 0.2, 0.3, inf, 0.5

            ================================================================

            compress: the value we want to have replaced with its number of occurrences
            mark: the marker value to use which is not of the vector data type, type to know a counter has been reached

        """
        res = [len(vector)]
        counter = 0
        for i, value in enumerate(vector):
            if value == compress:
                counter += 1
                continue
            elif counter > 2:
                res.append(np.NAN)
                res.append(counter)
            elif counter == 2:
                res.append(compress)
                res.append(compress)
            elif counter == 1:
                res.append(compress)
            res.append(value)
            counter = 0
        # so that if the vector ends in compress values, they wouldn't be lost
        if counter > 0:
            res.append(np.NAN)
            res.append(counter)

        return np.array(res, dtype=np.float64)


    def matrix_compressor(matrix, compress=np.inf):
        """compresses a martix by compresing each of the vectors in it

        Args:
            matrix (_type_): the matrix to compress
            compress (_type_, optional): value to compress based on of which is a lot. Defaults to np.inf.

        Returns:
            _type_: compressed matrix
        """
        res = []
        for i , vec in enumerate(matrix):
            res.append(Compress.vector_compressor(vec, compress))
        return np.array(res, dtype=object)


    def vector_decompressor(compressed_vector, replace_with=np.inf, final_type=np.float64):
        """decompresses a vector after it was compressed setting the vlaues where they blong adn with the amount they originally existed

        Args:
            compressed_vector (_type_): the vector to decompress
            replace_with (_type_, optional): replace the repeated value with. Defaults to np.inf.
            final_type (_type_, optional): the data type when returning the decompressed vector. Defaults to np.float64.

        Returns:
            _type_: the decompressed vector as it first was
        """
        res = []
        found_mark = False
        for i in range(1, len(compressed_vector)):
            if found_mark:
                found_mark = False
                continue
            if np.isnan(compressed_vector[i]):
                found_mark = True
                count = int(compressed_vector[i+1])
                for j in range(count):
                    res.append(replace_with)
            else:
                res.append(compressed_vector[i])
        return np.array(res, dtype=final_type)


    def matrix_decompressor(compressed_matrix, replace_with=np.inf, final_type=np.float64):
        """decompresses a matrix after it was compressed setting the vlaues where they blong and with the amount they originally existed

        Args:
            compressed_matrix (_type_): the matrix to decompress
            replace_with (_type_, optional): replace the repeated value with. Defaults to np.inf.
            final_type (_type_, optional): the data type of values when returning the decompressed matrix. Defaults to np.float64.

        Returns:
            _type_: the decompressed matrix
        """
        res = []
        for i, vector in enumerate(compressed_matrix):
            res.append(Compress.vector_decompressor(vector, replace_with, final_type))
        return np.array(res, dtype=final_type)
    

    
def main_phrase_counter(lst):
    stemmed = []
    stmr = SS("english")
    for phrase in lst:
        for word in phrase.split():
            stemmed.append(stmr.stem(word))
    counts = []
    for i, val in enumerate(set(stemmed)):
        counts.append([stemmed.count(val), val])
    return sorted(list(counts), key= lambda x : x[0], reverse=True)[0:30]


def clust_subj_vec(mainPhrases, subj_list):
    vector = np.ndarray((len(subj_list),))
    for i, val in enumerate(subj_list):
        vector[i] = len(re.findall(f"{val[1]}.?", mainPhrases))
    return vector