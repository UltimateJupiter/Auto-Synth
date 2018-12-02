
import Configs.C_DX10_Random_15params_10k as config
from Configs.gen import single_generate

import time
import numpy as np
import keras
import pickle
import tensorflow
from datetime import datetime

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv1D, Conv2D, LSTM, Recurrent, MaxPool1D, MaxPool2D, Flatten
from keras import losses, metrics, optimizers

from keras.callbacks import EarlyStopping, TensorBoard
import Configs.General_Configs as gConfigs
from pickle_decode import decode as read_dataset

size = 20
test_params, test_wavs = read_dataset(config.test_set, config.conv_friendly)
test_params, test_wavs = test_params[:size], test_wavs[:size]

model_name = "./train_logs/DX10-Random-15params-10k-model-_991556.h5"
save_dir = model_name.replace("h5",".results/")

import os
from tqdm import tqdm
if os.path.isdir(save_dir):
    pass
else:
    os.mkdir(save_dir)

model = load_model(model_name)
pred_params = model.predict(test_wavs)
print(pred_params)

for i in tqdm(range(size)):
    fl_orig = save_dir + "{}-raw.wav".format(i + 1)
    fl_pred = save_dir + "{}-pred.wav".format(i + 1)
    fl_cmp = save_dir + "{}-compare.txt".format(i + 1)

    single_generate(test_params[i], fl_orig)
    single_generate(pred_params[i], fl_pred)
    fl_tmp = open(fl_cmp, "w")
    fl_tmp.write(str(test_params[i]) + "\n")
    fl_tmp.write(str(pred_params[i]) + "\n")
    fl_tmp.write(str(test_params[i] - pred_params[i]) + "\n")
