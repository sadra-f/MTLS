from pathlib import Path
import re
from datetime import datetime

def docs_of_pattern(dir:Path, pattern="*.txt", recursive=True):
    """returns a list of the files in the given directory that match the given pattern

    Args:
        dir (Path): the directory in which to search for docs
        pattern (str, optional): the file pattern to look for. Defaults to "*.txt".
        recursive (bool, optional): search only in the dir or the directories inside the dir aw. Defaults to True.

    Returns:
        tuple: a tuple with the list of file paths and the count of the number of files
    """
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
    """Reads through the ground truth files and returns their content as tuples 

    Args:
        dir (Path): the path in which to search for the files

    Returns:
        _type_: a list of tuples each containing the text and the date of the text
    """
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