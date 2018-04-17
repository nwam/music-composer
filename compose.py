import sys
import numpy as np
from keras.models import load_model
import smidi

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: python {} <model_name> <song_size>'.format(sys.argv[0]))
        exit()
    model_name = sys.argv[1]
    song_size = int(sys.argv[2])

    try:
        model = load_model(model_name)
    except:
        print('Failed to load model')
        exit()

    song = np.zeros(( song_size, smidi.NUM_MIDI_PITCHES ))
    L = 256
    x = np.random.rand( 1, L, smidi.NUM_MIDI_PITCHES )*2-1

    for i in range(song_size):
        y = model.predict(x)
        song[i] = y[0, -1].reshape(( smidi.NUM_MIDI_PITCHES, ))
        x = np.roll(x, -1, axis=1)
        x = y
    
    # TODO: generate midi instead of plotting
    import matplotlib.pyplot as plt
    plt.imshow(song)
    plt.show()
