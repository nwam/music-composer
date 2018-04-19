import sys
import numpy as np
from keras.models import load_model
import smidi
from hyperps import n_features_in, n_features_out, L
import dataset

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

    data = dataset.load('banjo')
    x = data[40000:44000]

    song = np.zeros(( song_size, n_features_out))

    for i in range(song_size):
        y = model.predict(np.expand_dims(x[i:i+L],0))
        song[i] = y

    # TODO: generate midi instead of plotting
    import matplotlib.pyplot as plt
    plt.imshow(song)
    plt.show()
