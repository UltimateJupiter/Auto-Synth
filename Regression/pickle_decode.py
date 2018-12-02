import pickle
import numpy as np
import Configs.General_Configs as Configs

def decode(folder, expand_axis):
    
    params_fl = open(Configs.dataset_dir + "{}/{}-params.pkl".format(folder, folder), "rb")
    wavs_fl = open(Configs.dataset_dir + "{}/{}-wav.pkl".format(folder, folder), "rb")
    
    params = pickle.load(params_fl)
    wavs = pickle.load(wavs_fl)
    
    params, wavs = np.array(params), np.array(wavs)

    if expand_axis:
        wavs = np.expand_dims(wavs, -1)
    
    print(wavs.shape)
    params_fl.close()
    wavs_fl.close()

    return params, wavs
