from datetime import date as Date
from pathlib import Path
from models.TStr import TStr


class Document:
    def __init__(self, text:list[TStr], date:Date=None, path:(str|Path)=None, return_lowercase:bool=True) -> None:
        self._text = text
        self.return_lowercase = return_lowercase
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
    def text(self) -> list[TStr]:
        return [val.lower() for val in self._text] if self.return_lowercase else self._text
    
    @property
    def date(self):
        return self._date
    
    @date.setter
    def date(self, value:Date):
        self._date = value

    @date.getter
    def date(self) -> Date:
        return self._date

    @property
    def path(self):
        return self._path

