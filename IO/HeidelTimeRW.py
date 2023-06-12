from statics.paths import HT_LOG_PATH
from statics.config import DO_EXEC_LOG
import sys
from datetime import datetime
import re
from pathlib import Path
from datetime import date as Date


class HeideltimeRW:
    path = HT_LOG_PATH
    FILE_PATTERN = "*.txt"
    def __init__(self):
        pass

    def write_one_file(ht_list:list[str], origin_file_path, doc_id, open_as='w'):
        meta_data = HeideltimeRW.Meta(origin_file_path, doc_id)
        file_path = Path(str(HeideltimeRW.path) + '\\' + re.subn("/|\\\\|:",'-', str(origin_file_path))[0])

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

    def _read_one_file(path):
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
            print(f"Finished reading heideltime from file: {path}")
            return (results, meta)
        except:
            if DO_EXEC_LOG:
                print(f"Error reading heideltime result strings from file {path}")
            return None
        
    def read_all(dir_path=path):
        results = []
        path_iterator = HeideltimeRW.path.glob(dir_path)
        file_list = []

        while True:
            try:
                file_list.append(next(path_iterator))
            except StopIteration:
                break

        for path in file_list:
            results.append(HeideltimeRW._read_one_file(path))
        return results

    def replicate_dataset(ds_path:Path, dataset_dir_name:str):
        
        def _init_log():
            with open(NEW_PATH_BASE+'.log', 'w') as file:
                print("dataset path :" + ds_path, file=file)
                print("dataset_dir_name :" + dataset_dir_name, file=file)
                print("found doc # :" + doc_count, file=file)
                print("init at :" + datetime.now(), file=file)
                print('================================================================', file=file)

        def _log(origin_file, destination_file, length, result:bool, append_log):
            with open(NEW_PATH_BASE+'.log', 'a') as file:
                print("ID :" + _log.counter, file=file)
                print("dataset file :" + origin_file, file=file)
                print("destination file :" + destination_file, file=file)
                print("length :" + length, file=file)
                print("Ended at :" + datetime.now(), file=file)
                print("result :" + ("success" if result else "failure"), file=file)
                print("appended log :\r\n" + append_log, file=file)
                print('================================================================', file=file)
                _log.counter += 1
        _log.counter = 0
        
        ds_path = ds_path.absolute()
        if re.search(f"{dataset_dir_name}$", str(ds_path)) is None:
            print("dataset_dir_name not at the end of ds_path ?! what are you doing ??")
            return False
        
        ds_path_layers = re.split("\\", str(ds_path)) # delete me
        ds_dir_name_indx = re.search(f"{dataset_dir_name}$", str(ds_path)).span()[1]
        
        NEW_PATH_BASE = "log\\HT\\"

        if ds_path is str: ds_path = Path(ds_path)

        from .helpers import docs_of_pattern
        all_docs, doc_count = docs_of_pattern(ds_path)

        # check with user ... or not up to u.. :)
        for doc in all_docs:
            print(str(doc))
        print(doc_count)
        if input("proceed? Y/N: ").lower() != ('y' or 'yes'):
            return False
        
        _init_log(len(all_docs))

        from TimeTagger.HeidelTime_Generator import ht
        for doc_path in all_docs:
            _log_txt = ""
            success = False
            try:
                new_doc_path = NEW_PATH_BASE + str(doc_path[ds_dir_name_indx:]).replace('.txt', '.htr')
                doc_text = []
                ht_res = []
                with open(doc_path) as inp_file:
                    doc_text = [line for line in inp_file]
                ht_res = ht(doc_text, date=Date.fromisoformat(Path(doc_path).parent.name))
                with open(new_doc_path, 'w') as op_file:
                    for value in ht_res:
                        print(value, file=op_file)
                success = True
            except:
                _log_txt += sys.exc_info() + '\r\n\r\n'
            finally:
                _log(doc_path, new_doc_path, len(doc_text), success, _log_txt)





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