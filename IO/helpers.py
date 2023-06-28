from pathlib import Path
import re
from datetime import datetime

def docs_of_pattern(dir:Path, pattern="*.txt", recursive=True):
    result = []
    counter = 0
    path_iterator = dir.rglob(pattern) if recursive else dir.glob(pattern)
    while True:
        try:
            result.append(str(next(path_iterator)))
            counter += 1
        except StopIteration:   
            break
    return (result, counter)


def read_ground_truth(dir:Path): # returns a list of date,text tuples
    gt_text = []
    with open(dir, 'r') as file:
        gt_text = file.readlines()

    res = []
    date = None
    text = ""
    for line in gt_text :
        if re.search("^-+$", line) is not None:
            res.append((date, text))
            date = None
            text = ""
            continue
        elif re.search("\d{4}-\d{2}-\d{2}", line) is not None:
            line = line.strip()
            date = datetime.strptime(line, "%Y-%m-%d")
            continue
        else:
            text += line + "\n"
            continue
    
    return res

read_ground_truth("C:\\Users\\TOP\\Desktop\\project\\mtl_dataset\\mtl_dataset\\L1\\D1\\groundtruth\\g")