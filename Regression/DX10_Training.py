import keras
import pickle
import tensorflow

from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, Conv2D, LSTM, Recurrent

import Configs.C_DX10_Random_15params_1k as config
from pickle_decode import decode as read_dataset

train_params, train_wavs = read_dataset(config.training_set)
test_params, test_wavs = read_dataset(config.test_set)
val_params, val_wavs = read_dataset(config.validation_set)
