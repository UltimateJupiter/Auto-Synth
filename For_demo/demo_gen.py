import numpy as np
# from librosa.feature import mfcc
from python_speech_features import mfcc
from fft_decompose import fft
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

def plotting(spec1, spec2, i, distance):

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=[8,8/16*9])

    c = ax0.pcolor(spec1, cmap="hot")
    ax0.set_title('Original Spectrogram')
    ax0.set_xlabel("Frequency")
    ax0.set_ylabel("Intensity (in log)")

    c = ax1.pcolor(spec2, cmap="hot")
    ax1.set_title('Predicted Spectrogram - MFCC distance: {}'.format(str(distance)[:5]))
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Intensity (in log)")

    fig.tight_layout()
    plt.savefig("{}-spec.png".format(i), dpi=300)
    plt.close()

fl_ls = []

for x in os.listdir("../Demo"):
    if not x.__contains__("-"):
        continue
    fl_ls.append(x.split("-")[0])

fl_ls = list(set(fl_ls))

blank_large = np.zeros(32050)
blank_small = np.zeros(10000)
sounds = []
for x in fl_ls:
    
    fl_raw = "../Demo/{}-raw.wav".format(x)
    _, wav_raw = wavfile.read(fl_raw)
    wav_raw = wav_raw.astype("float")
    sounds.append(wav_raw)
    sounds.append(blank_small)
    spec_raw = fft(wav_raw, 28) + 1

    fl_pred = "../Demo/{}-pred.wav".format(x)
    _, wav_pred = wavfile.read(fl_pred)
    wav_pred = wav_pred.astype("float")
    sounds.append(wav_pred)
    sounds.append(blank_large)
    spec_pred = fft(wav_pred, 28) + 1
    
    # print(wav_pred, wav_raw)

    r_mfcc = np.array(mfcc(wav_raw, winstep=0.05))
    p_mfcc = np.array(mfcc(wav_pred, winstep=0.05))

    res = np.array(r_mfcc - p_mfcc)
    print(res.shape)
    spec_pred, spec_raw = np.log2(spec_pred), np.log2(spec_raw)

    ls = [np.sqrt(v.dot(v)) for v in res]
    ret = np.mean(ls)
    print(ret)

    plotting(spec_raw, spec_pred, x, ret)
# final = np.concatenate(sounds, axis=0)
# print(len(final))
# final = final/50000
# wavfile.write("demo.wav",44100,final)
