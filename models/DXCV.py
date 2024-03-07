import numpy as np

class DocumentxClusterVector:
    """DocumentXClusterVector
    """
    def __init__(self, doc_cluster_vector:np.ndarray=None, rep_sent_vector:np.ndarray=None):
        self.doc_cluster_vector = doc_cluster_vector
        self.rep_sent_vector = rep_sent_vector
    

    def _reset_vector(self):
        """reset vectors to zero values but keeps the dimentions
        """
        self.doc_cluster_vector = np.zeros_like(self.doc_cluster_vector)
        self.rep_sent_vector = np.zeros_like(self.rep_sent_vector)


    def _copy(self):
        """makes a copy of the object and returns the new DXCV obj

        Returns:
            DXCV: the new object which is a copy of the self obj
        """
        res = DocumentxClusterVector()
        res.doc_cluster_vector = self.doc_cluster_vector
        res.rep_sent_vector = self.rep_sent_vector
        return res
