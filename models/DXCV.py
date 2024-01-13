import numpy as np

class DocumentxClusterVector:
    def __init__(self, doc_cluster_vector:np.ndarray=None, rep_sent_vector:np.ndarray=None):
        self.doc_cluster_vector = doc_cluster_vector
        self.rep_sent_vector = rep_sent_vector
    

    def _reset_vector(self):
        self.doc_cluster_vector = np.zeros_like(self.doc_cluster_vector)
        self.rep_sent_vector = np.zeros_like(self.rep_sent_vector)


    def _copy(self):
        res = DocumentxClusterVector()
        res.doc_cluster_vector = self.doc_cluster_vector
        res.rep_sent_vector = self.rep_sent_vector
        return res
