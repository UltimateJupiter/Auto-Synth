base_dir = "/Users/jupiter/DUKE/C18_Fall/PHYS_136/Research/Auto-Synth/"
tmp_folder = base_dir + "TMPGEN/"
dataset_folder = base_dir + "Datasets/"
resources_dir = base_dir + "resources/"

dx_vst = resources_dir + "mda_DX10.vst"
# dx_vst = resources_dir = "Dexed.vst"

midi_fl = resources_dir + "midi_export.mid"
generator = resources_dir + "mrswatson"

param_num = 16
# param_num = 147
sample_num = 1000
thread_num = 10

noise_percentage = 1e-3 # In Amplitude
fft_frame = 28
