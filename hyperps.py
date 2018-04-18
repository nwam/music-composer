import smidi

L = 128 # 16th notes = 8 measures

n_notes = smidi.NUM_MIDI_PITCHES
note_out_dims = len(smidi.OParams)

n_features = n_notes*note_out_dims
