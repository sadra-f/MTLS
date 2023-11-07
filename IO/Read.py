import numpy as np


def read_np_array(path):
    """Raed a pickled/not pickled file containing the data in an numpy array

    Args:
        path (Path|str): the path in which to look for the files

    Returns:
        ndarray: the numpy array which was saved into the given file
    """
    return np.load(path, allow_pickle=True)
