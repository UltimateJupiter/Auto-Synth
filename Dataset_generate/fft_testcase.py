import sample_sound
import fft_decompose

s = sample_sound.get_sample()
b = fft_decompose.fft(s, 20, visualization=True)
