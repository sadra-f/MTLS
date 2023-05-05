from datetime import date as Date
from pathlib import Path
from ..models.Document import Document

class DocumentReader:
    def __init__(self, path:Path, file_pattern:str="*.txt", parent_as_date:bool=True, recursive:bool=True, to_lower:bool=True):
        self._path = path
        self.parent_as_date = parent_as_date
        self.to_lower = to_lower
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

    def _read_file(self, path:Path, parent_as_date:bool=False) -> Document:
        if path.is_file():
            with open(path) as file:
                parent_dir_name = path.parent.name
                return Document(file.read(), Date.fromisoformat(parent_dir_name) if parent_as_date else Date.today(), path, self.to_lower)
            
    def read_all(self) -> list[Document] | Document:
        if self._is_file:
            return self._read_file(self.file_path_list[0], self.parent_as_date)
        else:
            result = []
            for path in self.file_path_list:
                result.append(self._read_file(path, self.parent_as_date))
            return result