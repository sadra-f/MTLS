from sklearn.cluster._dbscan import DBSCAN
from models.ClusteredData import ClusteredData

def dbscan(inp_array, eps, min_samples):
    clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(inp_array)
    return ClusteredData(clusterer.labels_)