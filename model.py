import dataset
import smidi

import sys
import os
import numpy as np

from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential

MODEL_DIR = 'saved_models'

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('ERROR: Please supply a model name to the program')
        print('Usage: python {} <model_name>'.format(sys.argv[0]))
        exit()
    model_name = '{}.h5'.format(sys.argv[1])

    print('Loading dataset')
    x, y = dataset.load('banjo')

    n_notes = smidi.NUM_MIDI_PITCHES

    print('Building model')
    model = Sequential()
    model.add(LSTM(32, input_shape=(None, n_notes), 
                   dropout=0.2, 
                   recurrent_dropout=0.2, 
                   return_sequences=True))
    model.add(Dense(n_notes, activation='tanh'))

    print('Compiling model')
    model.compile(optimizer='adam',
                  loss='mse')

    print('Training model')
    model.fit(np.array([x[0]]), np.array([y[0]]),
              batch_size=32,
              epochs=30)

    print('Saving model')
    model.save(os.path.join(MODEL_DIR, model_name))
