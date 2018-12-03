import numpy as np

def get_sample():
    fl = open("./sample.pkl","rb")
    import pickle
    get = pickle.load(fl)
    return get
