from datetime import date as Date
from pathlib import Path

class Document:
    def __init__(self, text:str, date:Date=None, path:(str|Path)=None) -> None:
        self._text = text
        if date is None:
            self._date = Date.today()
        else:
            self._date = date
        if path is not None:
            if type(path) is str:
                self._path = Path(path)
            else:
                self._path = path
        

    @property
    def text(self):
        return self._text
    
    @property
    def date(self):
        return self._date
    
    @property
    def path(self):
        return self._path

