from datetime import date as Date
from pathlib import Path
from models.Document import Document
from models.TStr import TStr
from .helpers import docs_of_pattern

class DocumentReader:
    """
        DocumentReader Class used to read the content of a number of files and store the content using Document class objects.
        specifically used for reading the datasets
    """
    def __init__(self, path:Path, file_pattern:str="*.txt", parent_dir_as_date:bool=True, recursive:bool=True, to_lower:bool=True):
        self._path = path
        self.parent_as_date = parent_dir_as_date
        self.to_lower = to_lower
        self._is_directory = False
        self._is_file = False
        self._doc_count = -1
        self.file_path_list = [] # on windows systems will be list[WindowsPath]
        if self._path.is_dir():
            self.is_directory = True
            self.file_path_list, self._doc_count = docs_of_pattern(self._path,file_pattern, recursive)

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
                tmp_date = None
                try:
                    tmp_date = Date.fromisoformat(parent_dir_name)
                except ValueError as e:
                    print(f"Could not Parse Date {parent_dir_name} - parent dir name is not a valid date")
                finally:
                    return Document([TStr(line.lower()) if self.to_lower else TStr(line.lower()) for line in file], path, date=tmp_date if parent_as_date else None)
            
    def read_all(self) -> list[Document]:
        """Read all documents in the given directory

        Returns:
            list[Document]: reads through all documents in the given directory and returns their content in a list fo Document objects
        """
        result = []
        if self._is_file:
            result.append(self._read_file(self.file_path_list[0], self.parent_as_date))
        else:
            for path in self.file_path_list:
                result.append(self._read_file(Path(path), self.parent_as_date))
        return result
    
    def reset_counter(self):
        self._doc_counter = -1