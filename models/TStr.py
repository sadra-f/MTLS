import numpy as np

class TStr(str):
    """
        if you are to use any string class methods that alter the text,
        the method must be overridden here to return a TStr instance and not a string instance
    """
    @property
    def date(self):
        return self._date
    
    @date.setter
    def date(self, value):
        self._date = value

    @date.getter
    def date(self):
        return self._date
    
    @property
    def doc_path(self):
        return self._doc_path
    
    @doc_path.setter
    def doc_path(self, value):
        self._doc_path = value
    
    @doc_path.getter
    def doc_path(self):
        return self._doc_path
    
    @property
    def doc_id(self):
        return self._doc_id
    
    @doc_id.setter
    def doc_id(self, value):
        self._doc_id = value
    
    @doc_id.getter
    def doc_id(self):
        return self._doc_id
    
    @property
    def id(self):
        return self.id
    
    @id.setter
    def id(self, value):
        self._id = value
    
    @id.getter
    def id(self):
        return self._id
    
    @property
    def cluster(self):
        return self._cluster
    
    @cluster.setter
    def cluster(self, value):
        self._cluster = value
    
    @cluster.getter
    def cluster(self):
        return self._cluster
    
    @property
    def vector(self):
        return self._vector
    
    @vector.setter
    def vector(self, value):
        self._vector = value

    @vector.getter
    def vector(self):
        return self._vector
    
    def _reset_vector(self):
        self._vector = np.zeros_like(self._vector)

    def _copy(self):
        res = TStr(self)
        res.date = self._date
        res.doc_path = self._doc_path
        res.doc_id = self._doc_id
        res.vector = self._vector
        return res