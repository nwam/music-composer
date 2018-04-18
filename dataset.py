import smidi
import pickle
import os
import re
DATA_DIR = 'data'
PICKLE_DIR = os.path.join(DATA_DIR, 'pickles')

def generate(dirs=None, pickle_name='data'):
    '''
    Loads smidis of dirs

    Input:
        dirs is a list of the directories in data/ to use 
            None uses all of the directories
    '''
    data = []
    files = get_files(dirs)
    midi_file = re.compile('.*\.mid$')

    for filename in files:
        try:
            if midi_file.fullmatch(filename) is not None:
                data.append(smidi.midi2smidi(filename))
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
