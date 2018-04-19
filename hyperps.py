import smidi

L = 128 # 16th notes

n_notes = smidi.NUM_PITCHES
note_out_dims = len(smidi.OParams)
note_in_dims = len(smidi.IParams)

n_features_in = n_notes*note_out_dims+note_in_dims
n_features_out = n_notes*note_out_dims
