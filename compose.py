import sys
import numpy as np
from keras.models import load_model
import smidi
import pdb

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

    L              = model.get_layer(index= 0).get_config()['batch_input_shape'][1]
    n_features_in  = model.get_layer(index= 0).get_config()['batch_input_shape'][2]
    n_features_out = model.get_layer(index=-1).get_config()['units']

    song = np.zeros(( song_size, n_features_out))
    songp = np.zeros_like(song)

    # Lets try a non-random inital seed
#    import dataset
#    data = dataset.load('banjo')
#    x = data[:L]
#    x = np.expand_dims(x, 0)
    x = np.random.rand( 1, L, n_features_in)*2-1

    for i in range(song_size):
        yp = model.predict(x)

        r = np.random.normal(0.5, 0.05, yp.shape)
        y = (yp>r).astype(int)

        song[i] = y
        songp[i] = yp
        x = np.roll(x, 1, axis=1)
        x[0,1] = np.concatenate((y, np.expand_dims(smidi.next_beat_array(x[0,-2,-4:]), axis=0)), axis=1)

    # TODO: generate midi instead of plotting
    import matplotlib.pyplot as plt
    plt.imshow(songp)
    plt.xlabel('Note')
    plt.ylabel('Time (16th note)')
    plt.show()
    plt.imshow(song)
    plt.xlabel('Note')
    plt.ylabel('Time (16th note)')
    plt.show()
