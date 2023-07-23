from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from models.ClusteredData import ClusteredData
from statics.config import *
from pyclustering.utils.metric import distance_metric, type_metric
from helpers.distances import custom_sentence_distance
from pyclustering.cluster.kmeans import kmeans as km
import random
from models.ClusteredData import ClusteredData


def normal_kmeans(vectorized_data, n_clusters):
    res =  KMeans(n_clusters, n_init=80).fit(vectorized_data)
    return ClusteredData(res.labels_)

def custom_kmeans(input_matrix, initial_centers=None):
    dist_metric = distance_metric(metric_type=type_metric.USER_DEFINED, func=custom_sentence_distance)
    if initial_centers is None:
        initial_centers = 10
    if type(initial_centers) is int:
        initial_centers = [input_matrix[random.randint(0, len(input_matrix))] for k in range(initial_centers)]
    if type(initial_centers) is not (list or np.ndarray):
        raise TypeError(f"invalid type for initial_centers being {type(initial_centers)} must be list or numpy.ndarray")
    clstrr = km(input_matrix, initial_centers, metric=dist_metric)
    res =  clstrr.process()
    return res.get_clusters()