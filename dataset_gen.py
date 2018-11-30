import numpy as np
import multiprocessing
from scipy.io import wavfile
import json
import gen
from tqdm import tqdm

training_set_size = 1000
parameters_num = gen.param_num

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

    ram_array = np.random.rand(length, param_num)
    ram *= 100
    ram_array.astype(np.uint8)


def dataset_gen(training_set_size, parameters_num):
    """Generate the training datasets
    Args:
        the size of training set (how many samples)
        the number of parameters for each generation (e.g. 147 for Dexed)
    Return:
        1. wave matrix with shape(num_samples, frames)
        2. parameter matrix with shape(num_samples, num_params)
    """

    params = param_gen(training_set_size, parameters_num)
    total_mat, tmp_mat = None, None

    print("Start Generating Data")
    for i in tqdm(range(training_set_size)):
        
        # Generate the data
        # TODO: Maybe change this to multiprocessing to make the process faster.
        wav_data = generate(params[i], tmp_folder).reshape(1, parameters_num)
        
        if tmp_mat is None:
            tmp_mat = wav_data
        else:
            tmp_mat = np.concatenate((tmp_mat, wav_data), axis=0)

        if (i % 100 == 0 and i > 0):
            if total_mat is None:
                total_mat = tmp_mat
            else:
                total_mat = np.concatenate((total_mat, tmp_mat), axis = 0)
            tmp_mat = None
        
        print("input training set shape: {}".format(total_mat.shape))
        print("ground truth set shape: {}".format(params.shape))

        return total_mat, params
