import pdb

import dataset
import smidi
from hyperps import n_features_out, n_features_in, L

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
    data = dataset.load('banjo')
    song = data

    print('Building model')
    model = Sequential()
    model.add(LSTM(256, input_shape=(L, n_features_in), 
                   dropout=0.2, 
                   recurrent_dropout=0.2))
    model.add(Dense(n_features_out, activation='sigmoid'))

    print('Compiling model')
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy')

    print('Init callbacks')
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    #check_file = 'checkpoints/weights.{epoch:02d}-{loss:.2f}.h5'
    #checkpoint = ModelCheckpoint(check_file, monitor='loss', save_best_only=True, mode='min')
    check_file = 'checkpoints/e{:02d}.h5'
    batch_check_file = 'checkpoints/e{:02d}-b{}.h5'

    print('Training model')
    epochs = 50
    batch_size = 32
    batches = dataset.shuffled_batches(data, L, batch_size)
    for epoch in range(epochs):
        for i, (x,y) in enumerate(batches):
            model.train_on_batch(x, y)
            if i%50 == 0:
                print('Trained on batch {}, loss is {}'.format(i, model.evaluate(x,y,batch_size=batch_size, verbose=0)))
            if i%500 == 0:
                model.save(batch_check_file.format(epoch, i))
        model.save(check_file.format(epoch))

    model_path = os.path.join(MODEL_DIR, model_name)
    print('Saving model to {}'.format(model_path))
    model.save(model_path)

if __name__ == '__main__':
    main()
