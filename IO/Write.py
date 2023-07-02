from statics.paths import *
import numpy as np


def write_np_array(arr, path):
    np.save(path, arr,allow_pickle=True)