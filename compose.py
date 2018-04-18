import sys
import numpy as np
from keras.models import load_model
import smidi
from hyperps import n_features, L
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

    song = np.zeros(( song_size, n_features))
    songp = np.zeros_like(song)

    # Lets try a non-random inital seed
    import dataset
    data = dataset.load('banjo_multi')
    x = data[4]
    x = x[:L]
    x = np.expand_dims(x, 0)
    #x = np.random.rand( 1, L, n_features )*2-1

    for i in range(song_size):
        yp = model.predict(x)

        r = np.random.normal(0.5, 0.15, yp.shape)
        y = (yp>r).astype(int)

        song[i] = y
        songp[i] = yp
        x = np.roll(x, -1, axis=1)
        x[0,-1] = y

    # TODO: generate midi instead of plotting
    import matplotlib.pyplot as plt
    plt.imshow(songp)
    plt.show()
    plt.imshow(song)
    plt.show()
