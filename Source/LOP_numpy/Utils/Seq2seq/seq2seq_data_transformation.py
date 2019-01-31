def pitch_to_pc(pitch):
	# Pitch to pitch-class
	pc = pitch % 12 
	octave = pitch / 12
	return pc, octave

def pc_to_pitch(pc, octave):
	# Pitch to pitch-class
	pitch = octave*12 + pc
	return pitch

def orch_t_to_seq(orch_t, mapping):
	# list 
	# [
	#	[ instru, pitch ,octave ], ...
	# ]
	ind_orch_pitch = np.non_zeros(orch_t)
	N_instru = len(list(mapping.key()))
	N_pitch = 12
	N_octave = 11

	orch_seq_instru = []
	orch_seq_pc = []
	orch_seq_octave = []
	for orch_ind in ind_orch_pitch:
		for instrument_name, ranges in mapping.items():
	        if instrument_name == 'Piano':
	            continue
	        index_min = ranges['index_min']
	        index_max = ranges['index_max']
	        pitch_min = ranges['pitch_min']
	        pitch_max = ranges['pitch_max']
	        index_instru = ranges['index_instru']
	        if index_min <= orch_ind < index_max:
	        	pitch = index - index_min + pitch_min
	        	break

	    pc, octave = pitch_to_pc(pitch)
	   	new_instru = np.zeros(N_instru)
	   	new_instru[index_instru] = 1
	   	new_pitch = np.zeros(N_pitch)
	   	new_pitch[pc] = 1
	   	new_octave = np.zeros(N_octave)
	   	new_octave[octave] = 1
		orch_seq_instru.append(new_instru)
		orch_seq_pc.append(new_pitch)
		orch_seq_octave.append(new_octave)
	return orch_seq_instru, orch_seq_pc, orch_seq_octave

def orch_seq_to_t(orch_seq, mapping, N_orch):
	orch_t = np.zeros(N_orch)
	for note in orch_seq:
		instru, pc, octave = note
		pitch = pc_to_pitch(pc, octave)
		for instrument_name, ranges in mapping.items():
	        if instrument_name == 'Piano':
	            continue
	        index_min = ranges['index_min']
	        index_max = ranges['index_max']
	        pitch_min = ranges['pitch_min']
	        pitch_max = ranges['pitch_max']
	        index_instru = ranges['index_instru']
	        if index_instru == instru:
	        	index = index_min + pitch - pitch_min
	    orch_t[index] = 1
	return orch_seq