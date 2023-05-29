from statics.paths import HT_LOG_PATH
from statics.config import DO_EXEC_LOG
import sys
from datetime import datetime
import re

class HeideltimeRW:
    path = HT_LOG_PATH
    FILE_PATTERN = "*.txt"
    def __init__(self):
        pass

    def write(ht_list:list[str], origin_file_path, doc_id, open_as='w'):
        meta_data = HeideltimeRW.Meta(origin_file_path, doc_id)
        file_path = str(HeideltimeRW.path) + '/' + re.subn("/|\\\\",'-', str(origin_file_path))[0]

        if open_as != 'w' and open_as != 'a':
            raise ValueError("open_as must have either the value 'w' or 'a'")
        if DO_EXEC_LOG:
            print (f"Started writing heideltime strings to file at path {file_path}")

        try:
            with open(file_path, open_as) as file:
                print(meta_data,file=file)
                for i in range(len(ht_list)):
                    print(ht_list[i].replace('\n', ' ').replace('\r', ''), file=file)

            if DO_EXEC_LOG:
                print (f"finished writing heideltime strings to file at path {file_path}")
            return True
        except:
            if DO_EXEC_LOG:
                print("Error writing heideltime to file: " + str(sys.exc_info()))
            return False

    def _read_one(path):
        results = []
        meta = None

        if DO_EXEC_LOG:
            print(f"Reading heideltime from file: {path}")
        try:
            with open(path, 'r') as file:
                for line in file:
                    if meta is None:
                        meta = HeideltimeRW.Meta.obj_from_str(line)
                    else:
                        results.append(line)
            return (results, meta)
        except:
            if DO_EXEC_LOG:
                print(f"Error reading heideltime result strings from file {path}")
            return None
        
    def read_all():
        results = []
        path_iterator = HeideltimeRW.path.glob(HeideltimeRW.FILE_PATTERN)
        file_list = []

        while True:
            try:
                file_list.append(next(path_iterator))
            except StopIteration:
                break

        for path in file_list:
            results.append(HeideltimeRW._read_one(path))
        return results

    class Meta:
        def __init__(self, doc_path, doc_id):
            self.doc_path = doc_path
            self.doc_id = doc_id
            self.datetime = datetime.now()
        
        def obj_from_str(string:str):
            values = string.split(',', 2)
            res =  HeideltimeRW.Meta(values[2], values[1])
            res.datetime = datetime.strptime(values[0], '%Y-%m-%d %H:%M:%S.%f')
            return res
        
        def __str__(self) -> str:
            return f"{str(self.datetime)},{self.doc_id},{self.doc_path}"