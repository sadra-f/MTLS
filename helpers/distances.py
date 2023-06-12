from datetime import date
from statics.config import MTLS_DISTANCE_CONST, TIME_DISTANCE_CONST, CLUSTER_DISTANCE_CONST
import numpy as np
from numpy import ndarray, dot
from models.TStr import TStr

def date_day_diff(date1:date, date2:date):
    return abs((date2 - date1).days)

def normalized_date_diff(date1:date, date2:date):
    return 1 - pow(TIME_DISTANCE_CONST, date_day_diff(date1, date2))

def vector_length(vector:ndarray):
    return np.sqrt(vector.dot(vector))

def cosine_distance(vector1:ndarray, vector2:ndarray):
    #to get cosine similarity remove the "1 - ..." or subtract 1 from the result and get the absolute value
    return 1 - cosine_similarity(vector1, vector2)

def cosine_similarity(vector1:ndarray, vector2:ndarray):
    return dot(vector1, vector2) / ( vector_length(vector1) * vector_length(vector2) )

def sentence_distance(vector1:ndarray, date1:date, vector2:ndarray, date2:date):
    return MTLS_DISTANCE_CONST * normalized_date_diff(date1, date2) + (1 - MTLS_DISTANCE_CONST) * cosine_distance(vector1, vector2)

def cluster_distance(cluster_vector_1:ndarray, sent1, cluster_vector_2:ndarray, sent2):
    return CLUSTER_DISTANCE_CONST * cosine_distance(sent1, sent2) + (1 - CLUSTER_DISTANCE_CONST) * cosine_distance(cluster_vector_1, cluster_vector_2)