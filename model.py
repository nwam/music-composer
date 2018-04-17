import dataset
import smidi

import sys
import os
import numpy as np

from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

MODEL_DIR = 'saved_models'

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('ERROR: Please supply a model name to the program')
        print('Usage: python {} <model_name>'.format(sys.argv[0]))
        exit()
    model_name = '{}.h5'.format(sys.argv[1])

    print('Loading dataset')
    x = dataset.load('banjo')

    n_notes = smidi.NUM_MIDI_PITCHES

    print('Building model')
    model = Sequential()
    model.add(LSTM(32, input_shape=(None, n_notes), 
                   dropout=0.2, 
                   recurrent_dropout=0.2, 
                   return_sequences=True))
    model.add(Dense(n_notes, activation='tanh'))


    print('Compiling model')
    optimizer = RMSprop(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')

    print('Preparing data')
    x = x[0] # just using one song for now
    L = 256 # Length of mini-batches
    N = len(x) - 1 - L # Number of mini batches
    xx = np.zeros(( N, L, smidi.NUM_MIDI_PITCHES ))
    yy = np.zeros(( N, L, smidi.NUM_MIDI_PITCHES ))
    for i in range(N):
        xx[i] = x[i:i+L]
        yy[i] = x[i+1:i+1+L]

    print('Training model')
    model.fit(xx, yy,
              batch_size=32,
              epochs=10,
              shuffle=True)

    model_path = os.path.join(MODEL_DIR, model_name)
    print('Saving model to {}'.format(model_path))
    model.save(model_path)
