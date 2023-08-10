import numpy as np
class ClusteredData:
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
            for i in range(self.cluster_count):
                seperated.append([])
            for i in range(len(self.labels)):
                seperated[self.labels[i]].append(i)
            self.seperated = np.array(seperated, dtype=object)