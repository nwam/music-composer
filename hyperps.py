import smidi

L = 64 # 16th notes

n_notes = smidi.NUM_MIDI_PITCHES
note_out_dims = len(smidi.OParams)

n_features = n_notes*note_out_dims
