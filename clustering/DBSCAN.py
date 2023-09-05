from sklearn.cluster._dbscan import DBSCAN
from models.ClusteredData import ClusteredData

def dbscan(distance_matrix, eps, min_samples):
    """
        runs dbscan algorithem on given input. the input must be distance matrix of ones required points

        inp_array : the input distance matrix to run the algo over
        eps: epsilon value for the dbscan algorithem
        min_samples : minimum number of points needed around a point for it to be a centroid

        
        return : outputs the clustering result of the algo using the ClusteredData class
    """
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
    return ClusteredData(clusterer.labels_)