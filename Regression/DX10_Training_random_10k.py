import time
import numpy as np
import keras
import pickle
import tensorflow
from datetime import datetime

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Conv2D, LSTM, Recurrent, MaxPool1D, MaxPool2D, Flatten
from keras import losses, metrics, optimizers

from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler

import Configs.C_DX10_Random_15params_10k as config
from pickle_decode import decode as read_dataset

train_params, train_wavs, _ = read_dataset(config.training_set, config.conv_friendly)
test_params, test_wavs, _ = read_dataset(config.test_set, config.conv_friendly)
val_params, val_wavs, _ = read_dataset(config.validation_set, config.conv_friendly)

wav_size = train_wavs[0].shape[0]
param_size = config.n_param

print(wav_size, param_size)

def exp_decay(epoch):
   initial_lrate = 0.5
   k = 0.75
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate

lrate = LearningRateScheduler(exp_decay)


def LSTM_network_builder(wav_size):

    input_ori_wav = Input(shape=(wav_size, 1, ))
    conv_1 = Conv1D(64, kernel_size=4, strides=2, padding="valid")(input_ori_wav)
    pool_1 = MaxPool1D(pool_size=4, padding="valid")(conv_1)
    conv_2 = Conv1D(128, kernel_size=8, strides=2, padding="valid")(pool_1)
    pool_2 = MaxPool1D(pool_size=16, padding="valid")(conv_2)
    LSTM_1 = LSTM(32, dropout=0.05, recurrent_dropout=0.1, return_sequences=True)(pool_2)
    LSTM_2 = LSTM(16, dropout=0.05, recurrent_dropout=0.1, return_sequences=False)(LSTM_1)
    
    fc_1 = Dense(128, activation="relu")(LSTM_2)
    fc_2 = Dense(64, activation="tanh")(fc_1)
    output = Dense(param_size, activation="tanh")(fc_2)
    model = Model(inputs=[input_ori_wav], outputs=[output])
    model.summary()

    return model


def Conv1_network_builder(wav_size):

    input_ori_wav = Input(shape=(wav_size, 1, ))
    conv_1 = Conv1D(40, kernel_size=4, strides=2, padding="valid")(input_ori_wav)
    pool_1 = MaxPool1D(pool_size=2, padding="valid")(conv_1)
    conv_2 = Conv1D(80, kernel_size=10, strides=2, padding="valid")(pool_1)
    pool_2 = MaxPool1D(pool_size=2, padding="valid")(conv_2)
    f_1 = Flatten()(pool_2)
    fc_1 = Dense(100, activation="relu")(f_1)
    fc_2 = Dense(80, activation="tanh")(fc_1)
    fc_3 = Dense(30, activation="tanh")(fc_2)
    output = Dense(param_size, activation="tanh")(fc_3)
    model = Model(inputs=[input_ori_wav], outputs=[output])
    model.summary()
    return model

def FC_network_builder(wav_size):

    input_ori_wav = Input(shape=(wav_size,))
    l1 = Dense(1000, activation="relu")(input_ori_wav)
    h = Dense(400, activation="relu")(l1)
    h = Dense(200, activation="relu")(h)
    h = Dense(100, activation="relu")(h)
    h = Dense(50, activation="relu")(h)
    h = Dense(50, activation="relu")(h)
    l3 = Dense(25, activation="tanh")(h)
    output = Dense(param_size, activation="tanh")(l3)
    model = Model(inputs=[input_ori_wav], outputs=[output])
    return model

def train():
    
    # model = FC_network_builder(wav_size)
    # model = Conv1_network_builder(wav_size)
    model = LSTM_network_builder(wav_size)
    
    # model.compile(optimizer=optimizers.Adam(),
    #              loss=losses.mean_squared_error,
    #              metrics=[metrics.mse, metrics.mae]
    #              )
    
    model.compile(optimizer=optimizers.SGD(),
                  loss=losses.MSE,
                  metrics=[metrics.mse, metrics.mae]
                  )
    
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, mode="auto")
    
    time_stamp = "_" + str(datetime.now().microsecond)

    tfboard = TensorBoard(log_dir=config.log_dir + time_stamp,
                          write_graph=True,
                          update_freq='batch')

    print("\n\n=======================\nTensorboard Logs:\n{}\n========================\n\n".format("tensorboard --logdir={}".format(config.local_log_dir + time_stamp)))
    time.sleep(10)
    

    model.fit(x=train_wavs,
              y=train_params,
              batch_size=config.batch_size,
              epochs=config.max_epoches,
              validation_data=(val_wavs, val_params),
              callbacks=[earlystopping, tfboard, lrate]
              )
    model.save(config.local_log_dir + "-model-" + time_stamp + '.h5')
    score = model.evaluate(x=test_wavs,
                           y=test_params)

    print(score)

train()
