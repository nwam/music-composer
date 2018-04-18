import dataset
import smidi
from hyperps import n_features, L

import sys
import os
import numpy as np

from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras import optimizers
import keras.backend as K

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
    model.add(LSTM(64, input_shape=(x.shape[1:]), 
                   activation='sigmoid',
                   dropout=0.2, 
                   recurrent_dropout=0.2))
    model.add(Dense(y.shape[1], activation='sigmoid'))

    print('Compiling model')
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='mse')

    print('Training model')
    model.fit(x, y,
              batch_size=32,
              epochs=10,
              shuffle=True)

    model_path = os.path.join(MODEL_DIR, model_name)
    print('Saving model to {}'.format(model_path))
    model.save(model_path)
    

def nooff_loss(t, y):
    '''
    Loss function for smidi model with no note offset.
    Representation is simply a positive value for onsets,
    where the value denotes the time until the offset ie
    1 is a whole note, 0.25 is a quarter note, etc.

    It seems like we cannot create a differentiable loss
    function for this model
    '''
    # If a note is held for too short, it isn't played
    short = 1/64
    y = K.clip(y-short, 0, 1-K.epsilon())

    # Loss for missed notes
    false_positives = K.cast(K.greater(y-t,0), 'float32')
    fn_loss = K.sum(false_positives)
    
    # Loss for extra notes
    #false_negatives = t>0 and y==0
    #fp_loss = K.sum(1*false_positives)

    # Loss for note hold time
    #true_positives = t>0 and y>0
    #tp_loss = K.sum(true_positives*K.abs(K.log(t)-K.log(y)))

    return fn_loss #+ fp_loss + 0.1*tp_loss

if __name__ == '__main__':
    main()
