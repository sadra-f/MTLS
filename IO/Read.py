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


class DocumentReader:
    def __init__(self, path:Path, file_pattern:str="*.txt", parent_as_date:bool=True, recursive:bool=True):
        self._path = path
        self.parent_as_date = parent_as_date
        self._is_directory = False
        self._is_file = False
        self._doc_count = -1
        self.file_path_list = [] # on windows systems will be list[WindowsPath]
        if self._path.is_dir():
            self.is_directory = True
            path_iterator = self._path.rglob(file_pattern) if recursive else self._path.glob(file_pattern)
            while True:
                try:
                    self.file_path_list.append(next(path_iterator))
                    self._doc_count += 1
                except StopIteration:
                    break
        elif self._path.is_file():
            self.is_file = True
            self._doc_count = 1
            self.file_path_list.append(self._path)
        else:
            raise ValueError("Path must be a file or directory")
    
    @property
    def path(self):
        return self._path

    def _read_file(path:Path, parent_as_date:bool=False) -> Document:
        if path.is_file():
            with open(path) as file:
                parent_dir_name = path.parent.name
                return Document(file.read(), Date.fromisoformat(parent_dir_name) if parent_as_date else Date.today(), path)
            
    def read_all(self) -> list[Document] | Document:
        if self._is_file:
            return DocumentReader._read_file(self.file_path_list[0], self.parent_as_date)
        else:
            result = []
            for path in self.file_path_list:
                result.append(DocumentReader._read_file(path, self.parent_as_date))
            return result