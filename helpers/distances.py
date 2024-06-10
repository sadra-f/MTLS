from datetime import date
from statics.config import MTLS_DISTANCE_CONST, TIME_DISTANCE_CONST, CLUSTER_DISTANCE_CONST
import numpy as np
from numpy import ndarray, dot
from models.TStr import TStr
from models.DXCV import DocumentxClusterVector as DXCV 


def date_day_diff(date1:date, date2:date):
    """
        returns the absolute difference between two dates in days (int)
    """
    return abs((date2 - date1).days)

def normalized_date_diff(date1:date, date2:date):
    """
    returns the normalized difference between two dates for it to fit better into our equation (float)
    """
    return 1 - pow(TIME_DISTANCE_CONST, date_day_diff(date1, date2))

def vector_length(vector:ndarray):
    """
        retuns the mathmatical length of the vector (float)
    """
    return np.sqrt(vector.dot(vector))

def cosine_distance(vector1:ndarray, vector2:ndarray):
    """
        retuns the matmathmatical cosine distance of two vectors of the same number of features/dimensions (float)
    """
    #to get cosine similarity remove the "1 - ..." or subtract 1 from the result and get the absolute value
    res =  1 - cosine_similarity(vector1, vector2)
    return res if res > 0 else 0

def cosine_similarity(vector1:ndarray, vector2:ndarray):
    """
        returns the cosine similarity of two vectors (float)
    """
    return dot(vector1, vector2) / ( vector_length(vector1) * vector_length(vector2) )
    
def _sentence_distance(vector1:ndarray, date1:date, vector2:ndarray, date2:date):
    """
        returns the distance between senteces considering their vectorized representation and the date they refer to (float)
    """
    return MTLS_DISTANCE_CONST * normalized_date_diff(date1, date2) + (1 - MTLS_DISTANCE_CONST) * cosine_distance(vector1, vector2)


def _cluster_distance(cluster_vector_1:ndarray, sent1, subj1, cluster_vector_2:ndarray, sent2, subj2):
    """
        returns the distance between document vector of two clusters and the vector for the representing sentence of the cluster (float)
    """
    return 0.1 * cosine_distance(sent1, sent2) + 0.5 * cosine_distance(cluster_vector_1, cluster_vector_2) + 0.4 * cosine_distance(subj1, subj2)

def cluster_distance(dxcv1:DXCV, dxcv2:DXCV):
    return _cluster_distance(dxcv1.doc_cluster_vector, dxcv1.rep_sent_vector, dxcv1.subj_vector, dxcv2.doc_cluster_vector, dxcv2.rep_sent_vector, dxcv2.subj_vector)


def sentence_distance(sentence_data_1, sentence_data_2):
    """
        input TStr object containing vector of the sentence along the date it refers to
        
        returns the distance between senteces considering their vectorized representation and the date they refer to (float)
    """
    return _sentence_distance(sentence_data_1.vector, sentence_data_1.date, sentence_data_2.vector, sentence_data_2.date)