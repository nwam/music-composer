import pdb

import dataset
import smidi
from hyperps import n_features_out, n_features_in, L

import sys
import os
import numpy as np
import time
import math
import logging

from keras.layers import LSTM, Dense, Activation, Conv1D
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint

def create_model(model_name, dataset_name='banjo', weights_file=None):

    logging.basicConfig(filename='log.log', level=logging.DEBUG)

    MODEL_DIR = 'saved_models'

    model_name = '{}.h5'.format(model_name)

    print('Loading dataset')
    data = dataset.load(dataset_name)
    song = data

    print('Building model')
    model = Sequential()
#    model.add(Conv1D(32, 25, input_shape=(L, n_features_in),
#                     activation='sigmoid'))
#    model.add(Conv1D(32, 25,
#                     activation='sigmoid'))
    model.add(LSTM(256, input_shape=(L, n_features_in), 
                   dropout=0.2, 
                   recurrent_dropout=0.2,
                   activation = 'tanh'))
    model.add(Dense(n_features_out, activation='relu'))
    
    if weights_file is not None:
        print('Loading weights')
        model.load_weights(weights_file)

    print('Compiling model')
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')

    print('Init callbacks')
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    #check_file = 'checkpoints/weights.{epoch:02d}-{loss:.2f}.h5'
    #checkpoint = ModelCheckpoint(check_file, monitor='loss', save_best_only=True, mode='min')
    check_file = 'checkpoints/e{:02d}.h5'
    batch_check_file = 'checkpoints/e{:02d}-b{}-loss{}.h5'

    print('Training model')
    epochs = 20
    batch_size = 32
    min_loss = math.inf

    for epoch in range(epochs):
        batches = dataset.shuffled_batches(data, L, batch_size)
        for i, (x,y) in enumerate(batches):

            model.train_on_batch(x, y)

            if i%50 == 0:
                loss = model.evaluate(x,y,batch_size=batch_size, verbose=0)
                logging.info('Trained on batch {}-{}, loss is {}'.format(epoch, i, loss))
                print('Trained on batch {}-{}, loss is {}'.format(epoch, i, loss))
                if loss < min_loss: 
                    min_loss = loss
                    model.save(batch_check_file.format(epoch, i, min_loss))


        model.save(check_file.format(epoch))

    model_path = os.path.join(MODEL_DIR, model_name)
    print('Saving model to {}'.format(model_path))
    model.save(model_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('ERROR: Please supply a model name to the program')
        print('Usage: python {} <model_name> <dataset_name> <weights_file>'.format(sys.argv[0]))
        exit()

    dataset_name = None
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]

    weights_file = None
    if len(sys.argv) > 3:
        weights_file = sys.argv[3]


    create_model(sys.argv[1], dataset_name, weights_file=weights_file)
