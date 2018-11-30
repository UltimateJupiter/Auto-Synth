import numpy as np
import multiprocessing
from scipy.io import wavfile
import json

def param_gen(length, param_num):
    
    # A simple random version
    ram_array = np.random.rand(param_num)
    ram *= 100
    ram_array.astype(np.uint8)
