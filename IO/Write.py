from statics.paths import *
import numpy as np


def write_np_array(arr, path):
    """Writes a numpy array into a file, use pickling if needed

    Args:
        arr (ndarray): the numpy array to be written
        path (Path|str): the path into which ti write the array
    """
    np.save(path, arr, allow_pickle=True)