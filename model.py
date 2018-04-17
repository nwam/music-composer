import dataset
import smidi
from hyperps import n_notes, L

import sys
import os
import numpy as np

from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras import optimizers

MODEL_DIR = 'saved_models'

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('ERROR: Please supply a model name to the program')
        print('Usage: python {} <model_name>'.format(sys.argv[0]))
        exit()
    model_name = '{}.h5'.format(sys.argv[1])

    print('Loading dataset')
    data = dataset.load('banjo')

    print('Preparing data')
    data = data[0] # just using one song for now
    N = len(data) - 1 - L # Number of sequences
    x = np.zeros(( N, L, n_notes))
    y = np.zeros(( N, n_notes))
    for i in range(N):
        x[i] = data[i:i+L]
        y[i] = data[i+L]


    print('Building model')
    model = Sequential()
    model.add(LSTM(64, input_shape=(x.shape[1:]), 
                   activation='tanh',
                   dropout=0.2, 
                   recurrent_dropout=0.2))
    model.add(Dense(y.shape[1], activation='tanh'))


    print('Compiling model')
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='mse')

    print('Training model')
    model.fit(x, y,
              batch_size=32,
              epochs=20,
              shuffle=True)

    model_path = os.path.join(MODEL_DIR, model_name)
    print('Saving model to {}'.format(model_path))
    model.save(model_path)
