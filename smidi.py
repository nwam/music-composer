''' 
smidi - simple midi
    notes are snapped to beats based on midi's tempo
    and songs are represented in single arrays
'''

import pretty_midi
import numpy as np
from scipy.interpolate import interp1d
from enum import Enum

FILE_NAME = 'data/mario/Super Mario World 2 Yoshis Island - Story Music Box.mid'
NUM_MIDI_PITCHES  = 128
MAX_MIDI_VELOCITY = 127

class OParams(Enum):
    HOLD = 0
    PRESS = 1

class IParams(Enum):
    BEAT0 = 0
    BEAT1 = 1
    BEAT2 = 2
    BEAT3 = 3

def midi2smidi(filename, resolution=16, time_sig=4):
    '''
    Input:
        filename is the location of a midi file to parse
        resolution is the smallest beat step (default is 16th notes)
    '''
    
    # Load midi
    pm = pretty_midi.PrettyMIDI(filename)

    # Check time signature
    if time_sig is not None:
        sigs = pm.time_signature_changes
        for sig in sigs:
            if sig.numerator % time_sig != 0:
                raise TimeSignatureException('Time signature ({}/{}) on file {}'.format(sig.numberator, sig.denominator, filename))

    # Get timings of <resolution> beats (ie 32nd-note beats)
    beats4 = pm.get_beats()
    beats4 = np.append(beats4, 2*beats4[-1] - beats4[-2]) # extra beat to capture the last notes
    beat_interp = interp1d(np.arange(len(beats4)), beats4)
    interp_factor = resolution / 4
    beats = beat_interp(np.arange((len(beats4)-1)*interp_factor)/interp_factor)
    beats = np.append(beats, beats4[-1]) # last beat gets cut out in interp

    # Create a low resolution piano roll
    roll = np.zeros(( len(beats), NUM_MIDI_PITCHES, len(OParams) ))
    # Function to convert times to beat number
    time2beat = interp1d(beats, np.arange(len(beats)))

    # Put notes into low resolution roll
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            beat_s = int(np.round(time2beat(note.start)))
            beat_e = int(np.round(time2beat(note.end)))

            roll[beat_s][note.pitch][OParams.PRESS.value] = 1
            for beat in range(beat_s, beat_e+1):
                roll[beat][note.pitch][OParams.HOLD.value] = 1 #note.velocity/MAX_MIDI_VELOCIY

    # Reshape roll so that LSTM likes it
    s = roll.shape
    roll = roll.reshape(s[0], s[1]*s[2]) 

    # Get ids for each beat (only works for 4/4 time)
    downbeat0 = time2beat(pm.get_downbeats()[0])
    beats_per_measure = 4
    beat_array = np.zeros(( len(beats), beats_per_measure ))
    for i in range(beats_per_measure):
        beat_array[:, i] = np.arange(downbeat0, downbeat0+len(beats))//(2**i)%2

    data = np.concatenate((roll, beat_array), axis=1)
    return data

class TimeSignatureException(Exception):
    pass
