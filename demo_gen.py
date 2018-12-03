import numpy as np
from scipy.io import wavfile
import os

fl_ls = []

for x in os.listdir("Demo"):
    if not x.__contains__("-"):
        continue
    fl_ls.append(x.split("-")[0])

fl_ls = list(set(fl_ls))

blank_large = np.zeros(32050)
blank_small = np.zeros(10000)
sounds = []
for x in fl_ls:
    
    fl_raw = "./Demo/{}-raw.wav".format(x)
    _, wav_raw = wavfile.read(fl_raw)
    sounds.append(wav_raw)
    sounds.append(blank_small)

    fl_pred = "./Demo/{}-pred.wav".format(x)
    _, wav_pred = wavfile.read(fl_pred)
    sounds.append(wav_pred)
    sounds.append(blank_large)

final = np.concatenate(sounds, axis=0)
print(len(final))
final = final/50000
wavfile.write("demo.wav",44100,final)
