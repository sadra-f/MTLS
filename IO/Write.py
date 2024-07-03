from statics.paths import *
import numpy as np
from IO.helpers import join_paths


def write_np_array(arr, path, L, D):
    """Writes a numpy array into a file, use pickling if needed

    Args:
        arr (ndarray): the numpy array to be written
        path (Path|str): the path into which ti write the array
    """
    np.save(join_paths(path, L , D, "npy"), arr, allow_pickle=True)