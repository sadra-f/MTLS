import numpy as np

class ClusteredData:
    """
        A class that holds the clutersing labels and meta data on the result of clustering algorithms.
        Takes the labels list as input and calculates several other values using the labels list.

        Fields:
            labels: The labels from the result of the clustering
            has_outlier: True if there is an outlier with -1 value, false otherwise
            clusters: The values by which the clusters are represented e.g [0, 1, 2, 3]
            cluster_count: The number of clusters
            outlier_count: The number of outlier values in the result of clustering 
            cluster_memeber_count: The number of memebrs for each cluster e.g [10, 23, 8, 3, 20]
            seperated: Seperated labels by cluster into seperate lists each for a cluster
    """
    def __init__(self, labels, do_count_members=False, seperate=True):
        self.labels = labels
        try:
            min_cluster = labels.min()
            max_cluster = labels.max()
        except:
            min_cluster = -2
            max_cluster = -2
        self.has_outlier = True if min_cluster < 0 else False
        self.clusters = [i for i in range(min_cluster+1 if self.has_outlier else min_cluster, max_cluster+1)]
        self.cluster_count = len(self.clusters)
        if do_count_members:
            self.outlier_count = 0
            self.cluster_member_count = [0 for i in range(self.cluster_count)]
            for i in range(len(labels)):
                if labels[i] == -1:
                    self.outlier_count += 1
                else:
                    self.cluster_member_count[labels[i]] += 1
        if seperate:
            seperated = []
            if self.cluster_count > 1:
                for i in range(self.cluster_count):
                    seperated.append([])
                for i in range(len(self.labels)):
                    seperated[self.labels[i]].append(i)
                self.seperated = np.array(seperated, dtype=object)