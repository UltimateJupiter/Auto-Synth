import Config
import numpy as np
import multiprocessing as mp
from scipy.io import wavfile
import gen
import time
import os
from datetime import datetime
import pickle as pkl
from gen import single_generate
from tqdm import tqdm
from fft_decompose import fft

import json
from IPython import embed

training_set_size = Config.sample_num
parameters_num = Config.param_num

base_dir = Config.base_dir
test_dir = './'
tmp_folder = Config.tmp_folder
resources_dir = Config.resources_dir
# dx_vst = resources_dir + 'mda_DX10.vst'
dx_vst = Config.dx_vst
midi_fl = Config.midi_fl
generator = Config.generator


def param_gen(length, param_num): 
    
    # TODO: make a better version
    
    if param_num == 15:
        rand_array = np.random.rand(length, param_num)
    # A simple random version
    else:
        rand_array = np.random.rand(length, param_num)
        rand_array = rand_array * 100
        rand_array.astype(np.uint8)
        # make integer

    return rand_array


def dataset_genenerate(training_set_size, parameters_num, progress_mark=False, thread=-1, output=list()):
    """Generate the training datasets
    Args:
        the size of training set (how many samples)
        the number of parameters for each generation (e.g. 147 for Dexed)
    Return:
        1. wave matrix with shape(num_samples, frames)
        2. parameter matrix with shape(num_samples, num_params)
    """

    np.random.seed(thread + int(datetime.now().microsecond / 500))
    tmp_folder = Config.tmp_folder
    params = param_gen(training_set_size, parameters_num)
    total_mat, tmp_mat = 0, 0
    total_fft, tmp_fft = 0, 0
    
    if thread >= 0:
        print("Thread #{} Starts".format(thread))

    if progress_mark:
        time.sleep(1)
        print("\nStart Generating Data\nDisplaying progress for thread #0:")
        iterate = tqdm(range(training_set_size))
    else:
        iterate = range(training_set_size)

    for i in iterate:
        
        # Generate the data
        wav_data = single_generate(params[i], tmp_folder, thread)
        fft_data = np.expand_dims(fft(wav_data, Config.fft_frame), axis=0)

        wav_data = wav_data.reshape((1, wav_data.shape[0]))
        # print(wav_data.shape)
        if isinstance(tmp_mat, int):
            tmp_mat = wav_data
        else:
            tmp_mat = np.concatenate((tmp_mat, wav_data), axis=0)

        if isinstance(tmp_fft, int):
            tmp_fft = fft_data
        else:
            tmp_fft = np.concatenate((tmp_fft, fft_data), axis=0)

        if (i % 100 == 0 and i > 0):
            if isinstance(total_mat, int):
                total_mat = tmp_mat
            else:
                total_mat = np.concatenate((total_mat, tmp_mat), axis = 0)
            tmp_mat = 0
    
        if (i % 100 == 0 and i > 0):
            if isinstance(total_fft, int):
                total_fft = tmp_fft
            else:
                total_fft = np.concatenate((total_fft, tmp_fft), axis = 0)
            tmp_fft = 0
    
    if isinstance(total_mat, int):
        total_mat = tmp_mat
    elif not isinstance(tmp_mat, int):
        total_mat = np.concatenate((total_mat, tmp_mat), axis = 0)
    
    if isinstance(total_fft, int):
        total_fft = tmp_fft
    elif not isinstance(tmp_fft, int):
        total_fft = np.concatenate((total_fft, tmp_fft), axis = 0)
    
    
    if thread != -1:
        ret = [total_mat, params, total_fft]
        encode_name = "{}{}T-{}".format(tmp_folder, id(datetime.now()), thread)
        fl = open(encode_name, "wb")
        ret_encoded = pkl.dump(ret, fl)
        fl.close()
        while (not os.path.exists(encode_name)):
            time.sleep(0.1)
        output.put(encode_name)
        print("Thread #{} Completed".format(thread))
    else:
        return total_mat, params


def multithread_data_generating(training_set_size, parameters_num, num_threads):
    
    ret_queue = mp.Queue()
    assert(num_threads > 1), "Numthreads should be larger than 1"
    
    subtask_size = int(training_set_size / num_threads)
    subtasks_sizes = [subtask_size for thread in range(num_threads)]
    subtasks_sizes[0] += training_set_size - subtask_size * num_threads
    display = [False for thread in range(num_threads)]
    display[0] = True

    processes = [mp.Process(target=dataset_genenerate, args=(subtasks_sizes[thread], parameters_num, display[thread], thread, ret_queue)) for thread in range(num_threads)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    ret_mats = list()
    encoded_names = [ret_queue.get() for thread in range(num_threads)]
    for name in encoded_names:
        pkl_fl = open(name, "rb")
        ret_mats.append(pkl.load(pkl_fl))
        pkl_fl.close()
        os.system.__call__("rm {}".format(name))

    params = np.concatenate([ret_mats[i][1] for i in range(num_threads)], axis = 0)
    ffts = np.concatenate([ret_mats[i][2] for i in range(num_threads)], axis = 0)
    wavs = np.concatenate([ret_mats[i][0] for i in range(num_threads)], axis = 0)
    
    print("input wave training set shape: {}".format(wavs.shape))
    print("input spectrum training set shape: {}".format(ffts.shape))
    print("ground truth set shape: {}".format(params.shape))
    
    return wavs, params, ffts


def generate(name):

    wavs, params, spectrums = multithread_data_generating(Config.sample_num, Config.param_num, Config.thread_num)
    if os.path.isdir(Config.dataset_folder + name):
        print("Folder Exists, Please Remove It Or Choose Another Name")
        exit()
    os.system.__call__("mkdir {}{}".format(Config.dataset_folder, name))
    wavs_fl = open("{}{}/{}-wav.pkl".format(Config.dataset_folder, name, name), "wb")
    params_fl = open("{}{}/{}-params.pkl".format(Config.dataset_folder, name, name), "wb")
    spectrums_fl = open("{}{}/{}-spectrums.pkl".format(Config.dataset_folder, name, name), "wb")
    pkl.dump(wavs, wavs_fl)
    pkl.dump(params, params_fl)
    pkl.dump(spectrums, spectrums_fl)

