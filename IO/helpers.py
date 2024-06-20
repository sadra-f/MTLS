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



def print_2d_array(arr,location):
    with open(location, "w+") as file:
        for i, val in enumerate(arr):
            print(f"cluster {i}", file=file)
            for j, valj in enumerate(val):
                print(valj, end=' ', file=file)
                try:
                    print(valj.date, file=file)
                except:
                    print(file=file)

            print(file=file)
            print(file=file)

def print_1d_array(arr,location):
    with open(location, "w+") as file:
        for i, val in enumerate(arr):
            print(f"cluster {i}", file=file)
            print(val, file=file)
            print(file=file)
            



def join_paths(dir:Path, L, D, fex):
    dir.cwd().joinpath(f"L{L}D{D}.{fex}")
    return dir