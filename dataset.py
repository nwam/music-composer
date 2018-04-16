import smidi
import os

DATA_DIR = 'data'

def load(dirs=None):
    '''
    Loads smidis of dirs

    Input:
        dirs is a list of the directories in data/ to use 
            None uses all of the directories
    '''
    data = []

    if dirs is not None:
        dirs = [os.path.join(DATA_DIR, dir) for dir in dirs]

    for root, subdirs, files in os.walk(DATA_DIR):
        if dirs is not None and root not in dirs:
            continue

        for file in files:
            filename = os.path.join(root, file)

            try:
                data.append(smidi.midi2smidi(filename))
            except:
                continue

    return data
