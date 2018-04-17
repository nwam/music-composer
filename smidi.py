''' 
smidi - simple midi
    notes are snapped to beats based on midi's tempo
    and songs are represented in single arrays
'''

import pretty_midi
import numpy as np
from scipy.interpolate import interp1d

FILE_NAME = 'data/mario/Super Mario World 2 Yoshis Island - Story Music Box.mid'
NUM_MIDI_PITCHES  = 128
MAX_MIDI_VELOCITY = 127

def midi2smidi(filename, resolution=32):
    '''
    Input:
        filename is the location of a midi file to parse
        resolution is the smallest beat step (default is 32nd notes)
    '''

    # Load midi
    pm = pretty_midi.PrettyMIDI(filename)

    # Get timings of <resolution> beats (ie 32nd-note beats)
    beats4 = pm.get_beats()
    beats4 = np.append(beats4, 2*beats4[-1] - beats4[-2]) # extra beat to capture the last notes
    beat_interp = interp1d(np.arange(len(beats4)), beats4)
    interp_factor = resolution / 4
    beats = beat_interp(np.arange((len(beats4)-1)*interp_factor)/interp_factor)
    beats = np.append(beats, beats4[-1]) # last beat gets cut out in interp

    # Create a low resolution piano roll
    roll = np.zeros(( len(beats), NUM_MIDI_PITCHES ))
    # Function to convert times to beat number
    time2beat = interp1d(beats, np.arange(len(beats)))

    # Put notes into low resolution roll
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            beat_s = int(np.round(time2beat(note.start)))
            beat_e = int(np.round(time2beat(note.end)))

            roll[beat_s][note.pitch] = 1 #note.velocity/MAX_MIDI_VELOCITY
            roll[beat_e][note.pitch] = -1

    return roll
