from datetime import date as Date
from pathlib import Path

class Document:
    def __init__(self, text:str, date:Date=None, path:(str|Path)=None) -> None:
        self.text = text
        if date is None:
            self.date = Date.today()
        else:
            self.date = date
        if path is not None:
            if type(path) is str:
                path = Path(path)
            else:
                path = path
        

    @property
    def get_text(self):
        return self.text
    
    @property
    def get_date(self):
        return self.date
    
    @property
    def get_path(self):
        return self.path


class Document_Reader:
    def __init__(self, path:Path):
        self.path = path
        self._is_directory = False
        self._is_file = False
        self._doc_count = -1
        if self.path.is_dir():
            self.is_directory = True
            self.file_path_list = [] # on windows systems will be list[WindowsPath]
            path_iterator = self.path.glob("*.txt")
            while True:
                try:
                    self.file_path_list.append(next(path_iterator))
                    self._doc_count += 1
                except StopIteration:
                    break
        elif self.path.is_file():
            self.is_file = True
        else:
            raise ValueError("Path must be a file or directory")
    
    @property
    def get_path(self):
        return self.path