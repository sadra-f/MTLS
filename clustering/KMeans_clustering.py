import random
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from pyclustering.cluster.kmeans import kmeans as km
from pyclustering.utils.metric import distance_metric, type_metric

from models.TStr import TStr
from models.ClusteredData import ClusteredData

from helpers.distances import sentence_distance

def normal_kmeans(vectorized_data, n_clusters):
    res =  KMeans(n_clusters, n_init=80).fit(vectorized_data)
    return ClusteredData(res.labels_)

def custom_kmeans(input_matrix, initial_centers=None):
    dist_metric = distance_metric(metric_type=type_metric.USER_DEFINED, func=sentence_distance)
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
    def __init__(self, data:list[TStr], K = 10, step_count=20, distance_function=sentence_distance, label_wrapper=None):
        self.data = data
        self.init_centroids = [[data[value], 0] for i, value in enumerate(kmeans_plusplus(np.array([val.vector for val in data], np.float64), K, n_local_trials=25)[1])]
        self.centroids = self.init_centroids
        self.step_count = step_count
        self._shape = (len(data), len(self.centroids))
        self._current_distances = np.full(self._shape, np.inf)
        self._dist_func = distance_function

        self.labels = np.full(len(data), -1)

    def process(self):
        for i in range(self.step_count):
            self._run_one_cycle()
            self._find_new_centroids()
        return self


    def _run_one_cycle(self):
        for i, data in enumerate(self.data):
            for j, centroid in enumerate(self.centroids):
                self._current_distances[i, j] = self._dist_func(data, centroid[0])                
            clust_indx = np.argmin(self._current_distances.take(i, 0))
            self.labels[i] = clust_indx
            self.centroids[clust_indx][1] += 1

    def _find_new_centroids(self):
        self._set_to_means()
        for value in self.centroids:
            value[0].vector = np.divide(value[0].vector, value[1])
            value[1] = 0

    def _set_to_means(self):
        for val in self.centroids : val[0]._reset_vector()
        for i, val in enumerate(self.labels):
            self.centroids[val][0].vector = np.add(self.centroids[val][0].vector, self.data[i].vector)













