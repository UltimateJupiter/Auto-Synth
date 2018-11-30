import Config
import numpy as np
import multiprocessing as mp
from scipy.io import wavfile
import gen
from gen import single_generate
from tqdm import tqdm

import json
from IPython import embed

training_set_size = Config.sample_num
parameters_num = Config.param_num

test_dir = './'
tmp_folder = './TMPGEN/'
resources_dir = './resources/'
# dx_vst = resources_dir + 'mda_DX10.vst'
dx_vst = resources_dir + 'Dexed.vst'
midi_fl = resources_dir + 'midi_export.mid'
generator = resources_dir + 'mrswatson'

def param_gen(length, param_num): 
    # A simple random version
    # TODO: make a better version

    rand_array = np.random.rand(length, param_num)
    rand_array = rand_array * 100

    # make integer
    rand_array.astype(np.uint8)

    return rand_array

def dataset_genenerate(training_set_size, parameters_num, progress_mark=False, thread=-1):
    """Generate the training datasets
    Args:
        the size of training set (how many samples)
        the number of parameters for each generation (e.g. 147 for Dexed)
    Return:
        1. wave matrix with shape(num_samples, frames)
        2. parameter matrix with shape(num_samples, num_params)
    """

    tmp_folder = Config.tmp_folder
    params = param_gen(training_set_size, parameters_num)
    total_mat, tmp_mat = 0, 0
    
    if thread >= 0:
        print("Thread #{} Starts".format(thread))

    if progress_mark:
        print("Start Generating Data")
        iterate = tqdm(range(training_set_size))
    else iterate = range(training_set_size)

    for i in iterate:
        
        # Generate the data
        # TODO: Maybe change this to multiprocessing to make the process faster.
        
        wav_data = single_generate(params[i], tmp_folder)
        wav_data = wav_data.reshape((1, wav_data.shape[0]))
        # print(wav_data.shape)
        if isinstance(tmp_mat, int):
            tmp_mat = wav_data
        else:
            tmp_mat = np.concatenate((tmp_mat, wav_data), axis=0)

        if (i % 100 == 0 and i > 0):
            if isinstance(total_mat, int):
                total_mat = tmp_mat
            else:
                total_mat = np.concatenate((total_mat, tmp_mat), axis = 0)
            tmp_mat = 0
    
    if isinstance(total_mat, int):
        total_mat = tmp_mat
    elif not isinstance(tmp_mat, int):
        total_mat = np.concatenate((total_mat, tmp_mat), axis = 0)

    print("input training set shape: {}".format(total_mat.shape))
    print("ground truth set shape: {}".format(params.shape))

    return total_mat, params
