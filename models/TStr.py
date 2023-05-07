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

    def set_date(self, date):
        self._date = date


    def get_date(self)->Date:
        return self._date