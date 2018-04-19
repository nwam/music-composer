import pretty_midi
import smidi
import pickle
import os
import re
import numpy as np 
from hyperps import n_features_in, n_features_out

DATA_DIR = 'data'
PICKLE_DIR = os.path.join(DATA_DIR, 'pickles')
DUMP_SIZE = 600

def generate(dirs=None, pickle_name='data', accepted_keys=None):
    '''
    Loads smidis of dirs

    Input:
        dirs is a list of the directories in data/ to use 
            None uses all of the directories
        pickle_name is the name of the file which to save the generated data
        accepted_keys are the pretty_midi.KeySignature values to accept
            seeting accepted_keys will transpose every song to C maj/ A min
    '''
    data = np.zeros((0, n_features_in))
    files = get_files(dirs)
    midi_file = re.compile('.*\.mid$')

    for filename in files:
        if midi_file.fullmatch(filename) is None:
            continue

        try:
            pm = pretty_midi.PrettyMIDI(filename)
        except:
            continue

        if accepted_keys is not None:
            keys = pm.key_signature_changes
            
            # Check key
            if len(keys) == 0 or len(keys) > 1 or keys[0].key_number not in accepted_keys:
                continue

            # Transpose
            key = keys[0].key_number
            if key >= 12: # minor
                key = (key+3)%12 # convert to major
            transpose = get_transpose(key)

            for instrument in pm.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    note.pitch += transpose

        try:
            data = np.concatenate((data, smidi.midi2smidi(pm)))

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

def get_transpose(key):
    transpose = key%6
    transpose *= -1
    if key >= 6:
        transpose = 6+transpose
    return key

def pickle_filename(pickle_name):
    return '{}.pickle'.format(os.path.join(PICKLE_DIR, pickle_name))
