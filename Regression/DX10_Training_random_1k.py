import time
import keras
import pickle
import tensorflow
from datetime import datetime

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Conv2D, LSTM, Recurrent
from keras import losses, metrics, optimizers

from keras.callbacks import EarlyStopping, TensorBoard

import Configs.C_DX10_Random_15params_1k as config
from pickle_decode import decode as read_dataset

train_params, train_wavs = read_dataset(config.training_set)
test_params, test_wavs = read_dataset(config.test_set)
val_params, val_wavs = read_dataset(config.validation_set)

wav_size = train_wavs[0].shape[0]
param_size = config.n_param

print(wav_size, param_size)


def FC_network_builder(wav_size):

    input_ori_wav = Input(shape=(wav_size,))
    l1 = Dense(200, activation="relu")(input_ori_wav)
    h = Dense(50, activation="tanh")(l1)
    h = Dense(50, activation="tanh")(h)
    l3 = Dense(25, activation="tanh")(h)
    output = Dense(param_size, activation="tanh")(l3)
    model = Model(inputs=[input_ori_wav], outputs=[output])
    return model

def train():
    
    model = FC_network_builder(wav_size)
    
    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.mean_squared_error,
                  metrics=[metrics.mse]
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
              callbacks=[earlystopping, tfboard]
              )

    score = model.evaluate(x=test_wavs,
                           y=test_params)

    print(score)

train()
