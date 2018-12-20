import numpy as np

# Only for developing

def get_sample():
    fl = open("./sample.pkl","rb")
    import pickle
    get = pickle.load(fl)
    return get
