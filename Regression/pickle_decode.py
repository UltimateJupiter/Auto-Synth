import pickle
import numpy as np
import Configs.General_Configs as Configs

def decode(folder, expand_axis):
    
    params_fl = open(Configs.dataset_dir + "{}/{}-params.pkl".format(folder, folder), "rb")
    spectrums_fl = open(Configs.dataset_dir + "{}/{}-spectrums.pkl".format(folder, folder), "rb")
    wavs_fl = open(Configs.dataset_dir + "{}/{}-wav.pkl".format(folder, folder), "rb")
    
    params = pickle.load(params_fl)
    wavs = pickle.load(wavs_fl)
    spectrums = pickle.load(spectrums_fl)
    
    params, wavs, spectrums = np.array(params), np.array(wavs), np.array(spectrums)
    
    if expand_axis:
        wavs = np.expand_dims(wavs, -1)
        # spectrums = np.expand_dims(spectrums, -1)
    
    for i in [params, wavs, spectrums]:
        print(i.shape)

    params_fl.close()
    wavs_fl.close()
    spectrums_fl.close()

    return params, wavs, spectrums
