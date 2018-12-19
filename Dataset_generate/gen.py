import numpy as np
import os
import scipy
import Config
from scipy.io import wavfile
from datetime import datetime

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

def base_command(generator, vst, midi, param, flname):
    ret = "{} --channels 1 --quiet --plugin \"{}\" --midi-file {} {} --output \"{}\"".format(generator, vst, midi, param, flname)
    return ret


def single_generate(ind_array, tmp_folder, thread):
    """ Generate a single file
    Args:
        ind_array: an 1-D array with the length of the prameter set
    Returns:
        The wavefile in a numpy array of int16
    """
    assert len(ind_array) == Config.param_num, (len(ind_array), ind_array)

    param_set = ""
    for x in range(0, len(ind_array)):
        param_set += "--parameter {},{} ".format(str(x), ind_array[x])
    
    fl_name = tmp_folder + "T{}-{}.wav".format(str(thread), str(np.random.randint(0, 1e9)))
    cmd = base_command(generator, dx_vst, midi_fl, param_set, fl_name)
    os.system.__call__(cmd)

    fs, data = wavfile.read(fl_name)
    os.system.__call__("rm {}".format(fl_name))

    return data
