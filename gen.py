import numpy as np
import os
import scipy
from text_parsing import ret_parse
import get_cmd
from scipy.io import wavfile
from datetime import datetime

test_dir = './'
resources_dir = './resources/'
# dx_vst = resources_dir + 'mda_DX10.vst'
dx_vst = resources_dir + 'Dexed.vst'
midi_fl = resources_dir + 'midi_export.mid'
generator = resources_dir + 'mrswatson'
fl_name = 'text.wav'

tmp_folder = './TMPGEN/'

def generate(ind_array, tmp_folder):
    
    # ind_array = ret_parse()
    assert len(ind_array) == 147, (len(ind_array), ind_array)

    param_set = ""
    for x in range(1, len(ind_array) + 1):
        param_set += "--parameter {},{} ".format(str(x), ind_array[x - 1])
    
    fl_name = tmp_folder + str(datetime.now()) + str(np.random.randint(0, 1000)) + ".wav"
    cmd = get_cmd.base_command(generator, dx_vst, midi_fl, param_set, fl_name)
    os.system.__call__(cmd)

    fs, data = wavfile.read(fl_name)
    return [data, ind_array]


print("Done")
