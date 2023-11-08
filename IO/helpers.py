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
