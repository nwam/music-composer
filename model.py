import pdb

import dataset
import smidi
from hyperps import n_features, L

import sys
import os
import numpy as np
import time

from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint

MODEL_DIR = 'saved_models'

def main():
    if len(sys.argv) < 2:
        print('ERROR: Please supply a model name to the program')
        print('Usage: python {} <model_name>'.format(sys.argv[0]))
        exit()
    model_name = '{}.h5'.format(sys.argv[1])

    print('Loading dataset')
    data = dataset.load('banjo_multi')

    print('Preparing data')
    data = data[4] # just using one song for now
    N = len(data) - 1 - L # Number of sequences
    x = np.zeros(( N, L, n_features ))
    y = np.zeros(( N, n_features ))
    for i in range(N):
        x[i] = data[i:i+L]
        y[i] = data[i+L]

    print('Building model')
    model = Sequential()
    model.add(LSTM(256, input_shape=(x.shape[1:]), 
                   dropout=0.2, 
                   recurrent_dropout=0.2))
    model.add(Dense(y.shape[1], activation='sigmoid'))

    print('Compiling model')
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy')

    print('Init callbacks')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    check_file = 'checkpoints/weights.{epoch:02d}-{loss:.2f}.h5'
    checkpoint = ModelCheckpoint(check_file, monitor='loss', save_best_only=True, mode='min')


    print('Training model')
    model.fit(x, y,
              batch_size=32,
              epochs=30,
              shuffle=True,
              callbacks=[tensorboard, checkpoint])

    model_path = os.path.join(MODEL_DIR, model_name)
    print('Saving model to {}'.format(model_path))
    model.save(model_path)

if __name__ == '__main__':
    main()
