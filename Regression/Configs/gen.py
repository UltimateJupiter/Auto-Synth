import numpy as np
import os
import scipy
from scipy.io import wavfile
from datetime import datetime
import General_Configs

base_dir = General_Configs.base_dir

tmp_folder = base_dir + "TMPGEN/"
dataset_folder = base_dir + "Datasets/"
resources_dir = base_dir + "resources/"

dx_vst = resources_dir + "mda_DX10.vst"
# dx_vst = resources_dir = "Dexed.vst"

midi_fl = resources_dir + "midi_export.mid"
generator = resources_dir + "mrswatson"

def base_command(generator, vst, midi, param, flname):
    ret = "{} --channels 1 --quiet --plugin \"{}\" --midi-file {} {} --output \"{}\"".format(generator, vst, midi, param, flname)
    return ret


def single_generate(ind_array, file_name):
    """ Generate a single file
    Args:
        ind_array: an 1-D array with the length of the prameter set
    Returns:
        The wavefile in a numpy array of int16
    """

    param_set = ""
    for x in range(1, len(ind_array) + 1):
        param_set += "--parameter {},{} ".format(str(x), ind_array[x - 1])

    cmd = base_command(generator, dx_vst, midi_fl, param_set, file_name)
    os.system.__call__(cmd)
