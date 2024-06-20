import numpy as np
from pathlib import Path
import re
from datetime import datetime
from IO.helpers import join_paths
def read_np_array(path, L, D):
    """Raed a pickled/not pickled file containing the data in an numpy array

    Args:
        path (Path|str): the path in which to look for the files

    Returns:
        ndarray: the numpy array which was saved into the given file
    """
    return np.load(join_paths(path, L, D, ".npy"), allow_pickle=True)


def read_all_GTs(dataset_path:Path, N_TIMELINES):
    res = []
    for i in range(N_TIMELINES):
        gt_path = dataset_path / 'groundtruth' / f'g{i+1 if N_TIMELINES > 1 else ""}'
        res.append(read_ground_truth(gt_path))
    return res

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