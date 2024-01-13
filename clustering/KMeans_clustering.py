import random
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from pyclustering.cluster.kmeans import kmeans as km
from pyclustering.utils.metric import distance_metric, type_metric

from models.TStr import TStr
from models.DXCV import DocumentxClusterVector as DXCV
from models.ClusteredData import ClusteredData

from helpers.distances import sentence_distance, cluster_distance

def normal_kmeans(vectorized_data, n_clusters):
    """
        runs scikit learn kmeans over data with the given number of clusters

        vectorized_data: the list of point vectors to run the kmeans on
        n_clusters: the number of final clusters

        returns: clustered data labels and more using the ClusteredData class
    """
    res =  KMeans(n_clusters, init='k-means++').fit(vectorized_data)
    return ClusteredData(res.labels_)

def custom_kmeans(input_matrix, initial_centers=None):
    """
        runs the pyclustering kmeans while using custom distance function

        input_matrix: the input matrix containing the point vectors
        initial_centers: the initial centers for kmeans or the quantity required

        returns: the pyclustering, kmeans clustering results
    """
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
    """
        Custom implementation of Kmeans Algorithm so required distance function can be used
        uses kemans++ to find inital centers


        data: the vector list to run the kmeans algorithm over
        k: the number of centroids to run the kmeans algorithm with
        step_count: the number of times to run the kmeans algorithm
        distance_function: the distance function to use to calculate the distance of the given points
    """
    def __init__(self, data:list[TStr], K = 10, step_count=20, distance_function=sentence_distance):
        self.data = data
        self.K = K
        self.init_centroids = self._find_init_centroids(data)
        self.centroids = self.init_centroids
        self.step_count = step_count
        self._shape = (len(data), len(self.centroids))
        self._current_distances = np.full(self._shape, np.inf)
        self._dist_func = distance_function
        self.labels = np.full(len(data), -1)
    
    def process(self):
        """
            runs the kmeans algorithm on the dataset and returns self object to read the data from i.e (labels)

            returns: self
        """
        for i in range(self.step_count):
            self._run_one_cycle()
            self._find_new_centroids()
        return self


    def _run_one_cycle(self):
        """
            runs one cycle of kmeans algorithm calculates distances between 
            centroids and points and appoints each point to a cluster(centroid)
        """
        for i, data in enumerate(self.data):
            for j, centroid in enumerate(self.centroids):
                self._current_distances[i, j] = self._dist_func(data, centroid[0])                
            clust_indx = np.argmin(self._current_distances.take(i, 0))
            self.labels[i] = clust_indx
            self.centroids[clust_indx][1] += 1

    def _find_new_centroids(self):
        """
            finds the mean point in each cluster and sets it as the new centroid
        """
        self._set_to_sums()
        for value in self.centroids:
            value[0].vector = np.divide(value[0].vector, value[1])
            value[1] = 0

    def _set_to_sums(self):
        """
            calculates the sum of the vectors in each cluster
        """
        for val in self.centroids : val[0]._reset_vector()
        for i, val in enumerate(self.labels):
            self.centroids[val][0].vector = np.add(self.centroids[val][0].vector, self.data[i].vector)

    def _find_init_centroids(self, data):
        return [[data[value]._copy(), 0] for i, value in 
                enumerate(kmeans_plusplus(np.array([val.vector for val in data]), self.K, n_local_trials=25)[1])]
    



class AltCustomKMeans(CustomKMeans):
    def __init__(self, data:list[DXCV], K = 10, step_count=20, distance_function=cluster_distance):
        super().__init__(data, K, step_count, distance_function)
    
    def _find_new_centroids(self):
        self._set_to_sums()
        for value in self.centroids:
            value[0].doc_cluster_vector = np.divide(value[0].doc_cluster_vector, value[1])
            value[0].rep_sent_vector = np.divide(value[0].rep_sent_vector, value[1])
            value[1] = 0


    def _set_to_sums(self):
        for val in self.centroids : val[0]._reset_vector()
        for i, val in enumerate(self.labels):
            self.centroids[val][0].doc_cluster_vector = np.add(self.centroids[val][0].doc_cluster_vector, self.data[i].doc_cluster_vector)
            self.centroids[val][0].rep_sent_vector = np.add(self.centroids[val][0].rep_sent_vector, self.data[i].rep_sent_vector)


    def _find_init_centroids(self, data):
        return [[data[value]._copy(), 0] for i, value in 
                enumerate(kmeans_plusplus(np.array([val.doc_cluster_vector for val in data]), self.K, n_local_trials=25)[1])]
