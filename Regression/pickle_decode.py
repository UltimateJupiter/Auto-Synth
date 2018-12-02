import pickle
import Configs.General_Configs as Configs

def decode(folder):
    
    params_fl = open(Configs.dataset_dir + "{}/{}-params.pkl".format(folder, folder), "rb")
    wavs_fl = open(Configs.dataset_dir + "{}/{}-wav.pkl".format(folder, folder), "rb")
    
    params = pickle.load(params_fl)
    wavs = pickle.load(wavs_fl)
    
    params_fl.close()
    wavs_fl.close()

    return params, wavs
