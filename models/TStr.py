from datetime import date as Date

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