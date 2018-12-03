import time
import numpy as np
import keras
import pickle
import tensorflow
from datetime import datetime

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Conv2D, LSTM, Recurrent, MaxPool1D, MaxPool2D, Flatten, Concatenate, RNN, GRU, SimpleRNN
from keras import losses, metrics, optimizers

from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler

import Configs.C_DX10_Random_15params_10k_spec as config
from pickle_decode import decode as read_dataset

train_params, train_wavs, train_specs = read_dataset(config.training_set, config.conv_friendly)
test_params, test_wavs, test_specs = read_dataset(config.test_set, config.conv_friendly)
val_params, val_wavs, val_specs = read_dataset(config.validation_set, config.conv_friendly)

wav_size = train_wavs[0].shape[0]
param_size = config.n_param
spec_shape = train_specs.shape

print(wav_size, param_size)

def exp_decay(epoch):
   initial_lrate = 0.5
   k = 0.75
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate

lrate = LearningRateScheduler(exp_decay)

def Pure_rnn_spec_network_builder():

    input_spec = Input(shape=(spec_shape[1], spec_shape[2],))
    rnn_1 = SimpleRNN(384, return_sequences=False)(input_spec)
    rnn_2 = LSTM(768, return_sequences=False)(input_spec)
    c_1 = Concatenate()([rnn_1, rnn_2])
    fc_1 = Dense(64, activation="tanh")(c_1)
    fc_2 = Dense(32, activation="tanh")(fc_1)
    #f_1 = Flatten()(fc_2)
    output = Dense(param_size, activation="relu")(fc_2)
    model = Model(inputs=[input_spec], outputs=[output])
    model.summary()

    return model

def Pure_cnn_spec_network_builder():

    input_spec = Input(shape=(spec_shape[1], spec_shape[2], spec_shape[3],))
    conv_spec_1 = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding="same")(input_spec)
    pool_spec_1 = MaxPool2D(pool_size=(2, 2))(conv_spec_1)
    conv_spec_2 = Conv2D(64, kernel_size=(3, 5), strides=(1, 2), padding="same")(pool_spec_1)
    pool_spec_2 = MaxPool2D(pool_size=(2, 8))(conv_spec_2)
    conv_spec_3 = Conv2D(256, kernel_size=(3, 5), strides=(1, 2), padding="same")(pool_spec_2)
    pool_spec_3 = MaxPool2D(pool_size=(2, 2))(conv_spec_3)

    f_spec = Flatten()(pool_spec_3)
    fc_1 = Dense(64, activation="tanh")(f_spec)
    fc_2 = Dense(32, activation="tanh")(fc_1)
    output = Dense(param_size, activation="relu")(fc_2)
    model = Model(inputs=[input_spec], outputs=[output])
    model.summary()

    return model

def LSTM_spec_network_builder():

    # Conv on spec
    input_spec = Input(shape=(spec_shape[1], spec_shape[2], spec_shape[3], ))
    
    conv_spec_1 = Conv2D(16, kernel_size=(3, 5), strides=(1, 1), padding="same")(input_spec)
    pool_spec_1 = MaxPool2D(pool_size=(2, 2))(conv_spec_1)
    conv_spec_2 = Conv2D(32, kernel_size=(3, 5), strides=(1, 2), padding="same")(pool_spec_1)
    pool_spec_2 = MaxPool2D(pool_size=(2, 8))(conv_spec_2)
    conv_spec_3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 2), padding="same")(pool_spec_2)
    pool_spec_3 = MaxPool2D(pool_size=(1, 2))(conv_spec_3)
    
    f_spec = Flatten()(pool_spec_3)
    fc_spec_1 = Dense(256, activation="relu")(f_spec)

    input_ori_wav = Input(shape=(wav_size, 1, ))
    conv_1 = Conv1D(64, kernel_size=4, strides=2, padding="valid")(input_ori_wav)
    pool_1 = MaxPool1D(pool_size=4, padding="valid")(conv_1)
    conv_2 = Conv1D(128, kernel_size=8, strides=2, padding="valid")(pool_1)
    pool_2 = MaxPool1D(pool_size=16, padding="valid")(conv_2)
    LSTM_1 = LSTM(16, dropout=0.05, recurrent_dropout=0.1, return_sequences=True)(pool_2)
    LSTM_2 = LSTM(8, dropout=0.05, recurrent_dropout=0.1, return_sequences=False)(LSTM_1)
    
    fc_1 = Dense(128, activation="relu")(LSTM_2)

    concat_1 = Concatenate()([fc_1, fc_spec_1])

    fc_2 = Dense(64, activation="tanh")(concat_1)
    output = Dense(param_size, activation="relu")(fc_2)
    model = Model(inputs=[input_ori_wav, input_spec], outputs=[output])
    model.summary()

    return model


def train():
    
    # model = FC_network_builder(wav_size)
    # model = Conv1_network_builder(wav_size)
    # model = LSTM_spec_network_builder()
    # model = Pure_conv_network_builder()
    model = Pure_rnn_spec_network_builder()
    
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
    

    model.fit(x=train_specs,
              y=train_params,
              batch_size=config.batch_size,
              epochs=config.max_epoches,
              validation_data=(val_specs, val_params),
              callbacks=[earlystopping, tfboard]
              )

    model.save(config.local_log_dir + "-model-" + time_stamp + '.h5')
    score = model.evaluate(x=test_specs,
                           y=test_params)
    '''
    model.fit(x=[train_wavs, train_specs],
              y=train_params,
              batch_size=config.batch_size,
              epochs=config.max_epoches,
              validation_data=([val_wavs, val_specs], val_params),
              callbacks=[earlystopping, tfboard]
              )

    model.save(config.local_log_dir + "-model-" + time_stamp + '.h5')
    score = model.evaluate(x=[test_wavs, test_specs],
                           y=test_params)
    '''
    print(score)

train()
