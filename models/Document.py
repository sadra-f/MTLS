from datetime import date as Date
from pathlib import Path
from models.TStr import TStr


class Document:
    def __init__(self, text:list[TStr], path:(str|Path), date:Date=None) -> None:
        self._text = text
        if date is None:
            self._date = Date.today()
        else:# dont do this here !!!
            self._date = date
        if path is not None:
            if type(path) is str:
                self._path = Path(path)
            else:
                self._path = path
        self._set_text_date()
        for i in range(len(self._text)):
            self._text[i].doc_path = hash(self._path)

    @property
    def text(self):
        return self._text
        
    @text.setter
    def text(self, value:list[TStr]):
        self._text = value

    @text.getter
    def text(self) -> list[TStr]:
        return self._text


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


    def _set_text_date(self):
        for t in self._text:
            t.doc_path = self._path