class ClusteredData:
     def __init__(self, labels, do_count_members=False):
        self.labels = labels
        min_cluster = labels.min()
        max_cluster = labels.max()
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
