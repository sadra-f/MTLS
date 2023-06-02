from pathlib import Path

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