import numpy as np


def read_np_array(path):
    return np.load(path, allow_pickle=True)
