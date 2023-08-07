import random
import numpy as np
from sklearn.cluster import KMeans

from pyclustering.cluster.kmeans import kmeans as km
from pyclustering.utils.metric import distance_metric, type_metric

from models.TStr import TStr
from models.ClusteredData import ClusteredData

from ..helpers.distances import custom_sentence_distance

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


class CustomKMeans:
    def __init__(self, data:list[TStr], initial_centers, step_count, distance_function=custom_sentence_distance, label_wrapper=None):
        self.data = data
        if type(initial_centers) is int:
            self.centeroids = [data[random.randint(0, len(data))] for k in range(initial_centers)]
        else:
            self.centeroids = initial_centers
        self.step_count = step_count
        self._shape = (len(data), len(initial_centers))
        self._current_distances = np.full(self._shape, np.inf)
        self._dist_func = distance_function

        self.labels = np.full((len(data), 1), -1)

    def process(self):
        for i in range(self.step_count):
            self._run_one_cycle()
            self._find_new_centroids()


    def _run_one_cycle(self):
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                self._current_distances[i, j] = self._dist_func(self.data[i], self.centeroids[j])                
            self.labels[i] = np.argmax(self._current_distances.take(i, 0))

    def _find_new_centroids(self):
        self._calc_means()
        pass

    def _calc_means(self):
        pass













