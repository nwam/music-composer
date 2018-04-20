import sys
import numpy as np
from keras.models import load_model
import smidi
import pdb
import dataset
import matplotlib.pyplot as plt
import logging


logging.basicConfig(filename='trainlog.log', level=logging.DEBUG)
        
def compose(model_name, song_size=2000, thresh=0.5, song_seed=None, output_name='song.mid'):
    print('Loading model')
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

    print('Creating initial seed')
#    data = song_seed
#    if data is not None:
#        seed = 0# np.random.randint(0, len(data)-L)
#        x = data[seed:seed+L]
#        x = np.expand_dims(x, 0)
    if song_seed is not None:
        data = dataset.load(song_seed)
        seed = np.random.randint(0, len(data)-L)
        x = data[seed:seed+L]
        x = np.expand_dims(x, 0)
    else:
        x = np.random.rand( 1, L, n_features_in)*2-1

    print('Composing music')
    for i in range(song_size):
        if i%500==0:
            print('{} beats written'.format(i))

        yp = model.predict(x)

        #r = np.random.normal(thresh, 0.05, yp.shape)
        y = (yp>thresh).astype(int)

        song[i] = y
        songp[i] = yp

#        x = np.roll(x, 1, axis=1)
#        x[0,1] = np.concatenate((y, np.expand_dims(smidi.next_beat_array(x[0,-2,-4:]), axis=0)), axis=1)
        x = np.append(x,np.expand_dims(np.concatenate((y, np.expand_dims(smidi.next_beat_array(x[0,-1,-4:]), axis=0)), axis=1), axis=0), axis=1)[:,1:]

    print('Saving song to {}'.format(output_name))
    smidi.smidi2midi(song, filename=output_name)
    
    return song, songp

# TODO: modularize!
def imitate(model_name, song_size=2000, thresh=0.5, song_seed=None, output_name='song.mid'):
    print('Loading model')
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

    print('Creating initial seed')
    if song_seed is not None:
        data = dataset.load(song_seed)
        seed = np.random.randint(0, len(data)-song_size-L)
        x = data[seed:seed+song_size+L+1]
        x = np.expand_dims(x, 0)
    else:
        x = np.random.rand( 1, L, n_features_in)*2-1

    print('Composing music')
    for i in range(song_size):
        if i%500==0:
            print('{} beats written'.format(i))

        yp = model.predict(np.expand_dims(x[0, i:i+L], 0))

        #r = np.random.normal(thresh, 0.05, yp.shape)
        y = (yp>thresh).astype(int)

        song[i] = y
        songp[i] = yp

    print('Saving song to {}'.format(output_name))
    smidi.smidi2midi(song, filename=output_name)
    
    # Get some stats
    y = song
    t = x[0,L+1:,:176]
    print('Overall performance')
    print_stats(y,t)
    
    yr = y.reshape(y.shape[0], y.shape[1]//2, 2)
    tr = t.reshape(t.shape[0], t.shape[1]//2, 2)
    yhold = yr[:,:,smidi.OParams.HOLD.value]
    thold = tr[:,:,smidi.OParams.HOLD.value]
    ypress = yr[:,:,smidi.OParams.PRESS.value]
    tpress = tr[:,:,smidi.OParams.PRESS.value]
    print('Hold performance')
    print_stats(yhold,thold)
    print('Press performance')
    print_stats(ypress, tpress)

    return song, songp, t

def print_stats(y,t):
    noty = (y+1)%2
    nott = (t+1)%2
    print('Accuracy:\t{}/{}={}'.format(np.sum(y==t),np.size(t), np.sum(y==t)/np.size(t)))
    print('True Pos:\t{}/{}={}'.format(np.sum(y*t) ,np.sum(t), np.sum(y*t) /np.sum(t)))
    print('False Pos:\t{}/{}={}'.format(np.sum(y*nott), np.sum(nott), np.sum(y*nott)/np.sum(nott)))
    print('True Neg:\t{}/{}={}'.format(np.sum(noty*nott), np.sum(nott),np.sum(noty*nott)/ np.sum(nott)))
    logging.info('Accuracy:\t{}/{}={}'.format(np.sum(y==t),np.size(t), np.sum(y==t)/np.size(t)))
    logging.info('True Pos:\t{}/{}={}'.format(np.sum(y*t) ,np.sum(t), np.sum(y*t) /np.sum(t)))
    logging.info('False Pos:\t{}/{}={}'.format(np.sum(y*nott), np.sum(nott), np.sum(y*nott)/np.sum(nott)))
    logging.info('True Neg:\t{}/{}={}'.format(np.sum(noty*nott), np.sum(nott),np.sum(noty*nott)/ np.sum(nott)))
    logging.info('False Neg:\t{}/{}={}'.format(np.sum(noty*t), np.sum(t),np.sum(noty*t)/ np.sum(t)))

def plot_song(song):
    plt.imshow(song)
    plt.xlabel('Note')
    plt.ylabel('Time (16th note)')
    plt.show()
    

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: python {} <model_name> <song_size> [<thresh>]'.format(sys.argv[0]))
        exit()
    model_name = sys.argv[1]
    song_size = int(sys.argv[2])

    thresh = 0.5
    if len(sys.argv) > 3:
        thresh = float(sys.argv[3])

    song_seed = None
    if len(sys.argv) > 4:
        song_seed = sys.argv[4]

    song, songp = compose(model_name, song_size, thresh, song_seed, output_name='song.mid')
    plot_song(songp)
    plot_song(song)
