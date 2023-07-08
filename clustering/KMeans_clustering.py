from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from models.ClusteredData import ClusteredData
from statics.config import *

def kmeans(vectorized_data, n_clusters):
    res =  KMeans(n_clusters, n_init=80).fit(vectorized_data)
    return ClusteredData(res.labels_)
