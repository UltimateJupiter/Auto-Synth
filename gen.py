import numpy as np
import os
import scipy
import Config
from text_parsing import ret_parse
from scipy.io import wavfile
from datetime import datetime

test_dir = './'
resources_dir = './resources/'
# dx_vst = resources_dir + 'mda_DX10.vst'
dx_vst = resources_dir + 'Dexed.vst'
midi_fl = resources_dir + 'midi_export.mid'
generator = resources_dir + 'mrswatson'
# fl_name = 'test.wav'

tmp_folder = './TMPGEN/'

def base_command(generator, vst, midi, param, flname):
    ret = "{} --channels 1 --quiet --plugin \"{}\" --midi-file {} {} --output \"{}\"".format(generator, vst, midi, param, flname)
    return ret


def single_generate(ind_array, tmp_folder, thread=-1):
    """ Generate a single file
    Args:
        ind_array: an 1-D array with the length of the prameter set
    Returns:
        The wavefile in a numpy array of int16
    """
    
    # ind_array = ret_parse()
    assert len(ind_array) == Config.param_num, (len(ind_array), ind_array)

    param_set = ""
    for x in range(1, len(ind_array) + 1):
        param_set += "--parameter {},{} ".format(str(x), ind_array[x - 1])
    
    fl_name = tmp_folder + "T{}-{}.wav".format(str(thread), str(np.random.randint(0, 1e9))
    cmd = base_command(generator, dx_vst, midi_fl, param_set, fl_name)
    os.system.__call__(cmd)

    fs, data = wavfile.read(fl_name)
    os.system.__call__("rm {}".format(fl_name))

    return data
