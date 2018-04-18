import smidi
import pickle
import os
import re
import numpy as np 
from hyperps import n_features_in, n_features_out

DATA_DIR = 'data'
PICKLE_DIR = os.path.join(DATA_DIR, 'pickles')

def generate(dirs=None, pickle_name='data'):
    '''
    Loads smidis of dirs

    Input:
        dirs is a list of the directories in data/ to use 
            None uses all of the directories
    '''
    data = np.zeros((0, n_features_in))
    files = get_files(dirs)
    midi_file = re.compile('.*\.mid$')

    for filename in files:
        try:
            if midi_file.fullmatch(filename) is not None:
                    data = np.concatenate((data, smidi.midi2smidi(filename)))
        except smidi.TimeSignatureException:
            print('Warning: failed to add {} because of time signature'.format(filename))
            continue
        except Exception:
            print('Warning: failed to add {} for unknown reasons'.format(filename))
            continue

    pickle_file = pickle_filename(pickle_name)
    pickle.dump(data, open(pickle_file, 'wb'))


def load(pickle_name='data'):
    pickle_file = pickle_filename(pickle_name)
    return pickle.load(open(pickle_file, 'rb'))


def shuffled_batches(data, L, n):
    '''
    Creates a generator object which returns batches of sequences

    Input:
        data is the loaded data
        L is the length of each sequence
        n is the sequence count returned by each yield
    '''
    
    N = len(data) - 1 - L # Number of sequences
    if N < n:
        print('Warning: not enough data supplied to create sequences')
        return None

    # Generate random indices from which to gather data
    pool = np.arange(N)
    np.random.shuffle(pool)
    pool = pool[:N-N%n].reshape(N//n, n)

    # Return groups of sequences
    for drops in pool:
        yield get_sequences(data, L, drops)  

def get_sequences(data, L, ns):
    '''
    Returns a sample of sequences from a large loaded dataset

    Inputs:
        data is the loaded data
        L is the length of each sequence
        ns is the list of indices from which to gather sequences
    '''

    x = np.zeros(( len(ns), L, n_features_in ))
    y = np.zeros(( len(ns), n_features_out ))

    for i, n in enumerate(ns):
        x[i] = data[n:n+L]
        y[i] = data[n+L, :n_features_out]

    return x,y

def get_files(dirs=None):
    if dirs is not None:
        dirs = [os.path.join(DATA_DIR, dir) for dir in dirs]

    for root, subdirs, files in os.walk(DATA_DIR):
        if dirs is not None and root not in dirs:
            continue

        for file in files:
            filename = os.path.join(root, file)
            yield filename

def pickle_filename(pickle_name):
    return '{}.pickle'.format(os.path.join(PICKLE_DIR, pickle_name))
