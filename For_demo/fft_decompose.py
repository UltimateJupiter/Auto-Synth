import numpy as np
import matplotlib.pyplot as plt

def fft(sound, frames, visualization=False):
    
    window = int(len(sound)/frames)
    assert window >= 768
    total_array_get = 0
    hann = np.hanning(window)

    for frame in range(frames):

        start, end = frame * window, (frame + 1) * window
        audio = sound[start: end]
        audio = audio * hann
        fft_frame = np.fft.fft(audio, 768)
        fft_abs = np.abs(fft_frame)[:384]
        fft_abs = fft_abs.reshape(1, len(fft_abs))
        #fft_abs = np.log10(fft_abs)

        if total_array_get == 0:
            total_array = fft_abs
            total_array_get = 1
        else:
            total_array = np.concatenate((total_array, fft_abs), axis=0)
    
    if visualization:
        plt.imshow(np.log2(total_array))
        plt.show()
    
    return total_array
