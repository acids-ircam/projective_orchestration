#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import mido
from mido import MidiFile
from LOP_s2s.Database.simplify_instrumentation import get_instru_mapping
from LOP_s2s.Database.program_change_mapping import program_change_mapping


def from_pc_octave_to_pitch(pc, octave):
	pitch = octave * 12 + pc
	return pitch

def write_midi_seq(score, quantization, ticks_per_beat, write_path, tempo=80, metadata=None):
	# score is a list = [(pc(12), octave(11), intensity(128), instru_integer, repeat_flag(1))]
	#
	# First convert score into a dictionnary:
	# {'Instrument': list_events}

	_, instru_mapping_reverse = get_instru_mapping()

	dict_tracks = {'Piano': []}
	for instru_name in instru_mapping_reverse.values():
		dict_tracks[instru_name] = []

	for event in score:
		time, notes = event
		
		if len(notes) == 0:
			# Silence !!
			for instru_name in instru_mapping_reverse.values():
				if len(dict_tracks[instru_name])!=0:
					dict_tracks[instru_name].append((time, set()))

		new_events = {'Piano': set()}
		for instru_name in instru_mapping_reverse.values():
			new_events[instru_name] = set()

		for note in notes:
			instrument_index = note[3]
			if instrument_index == -1:
				instru_name = 'Piano'
			else:
				instru_name = instru_mapping_reverse[instrument_index]
			# Remove instrument info
			new_events[instru_name].add(note)
		
		for instru_name in instru_mapping_reverse.values():
			if len(new_events[instru_name])!=0:
				dict_tracks[instru_name].append((time, new_events[instru_name]))
		if len(new_events["Piano"])!=0:
			dict_tracks["Piano"].append((time, new_events["Piano"]))

	# Tempo
	microseconds_per_beat = mido.bpm2tempo(tempo)
	# Write a pianoroll in a midi file
	mid = MidiFile()
	mid.ticks_per_beat = ticks_per_beat

	# Create a metaTrack
	# Add a new track with the instrument name to the midi file
	track = mid.add_track("metatrack")
	# Not advised, tends to fuck up everyting
	if metadata:
		if "time_signature" in metadata.keys():
			track.append(mido.MetaMessage('time_signature', numerator=metadata["time_signature"][0], denominator=metadata["time_signature"][1], 
				clocks_per_click=metadata["time_signature"][2], notated_32nd_notes_per_beat=metadata["time_signature"][3], time=0))
		if "key_signature" in metadata.keys():
			track.append(mido.MetaMessage('key_signature', key=metadata["key_signature"], time=0))
	# Tempo
	track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))

	for instrument_name, track_list in dict_tracks.items():

		if len(track_list) == 0:
			continue

		# Add a new track with the instrument name to the midi file
		track = mid.add_track(instrument_name)
		
		# Not advised, tends to fuck up everyting
		if metadata:
			if "time_signature" in metadata.keys():
				track.append(mido.MetaMessage('time_signature', numerator=metadata["time_signature"][0], denominator=metadata["time_signature"][1], 
					clocks_per_click=metadata["time_signature"][2], notated_32nd_notes_per_beat=metadata["time_signature"][3], time=0))
			if "key_signature" in metadata.keys():
				track.append(mido.MetaMessage('key_signature', key=metadata["key_signature"], time=0))

		# Tempo
		track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
		
		# Add the program_change (for playback, GeneralMidi norm)
		program = program_change_mapping[instrument_name]-1
		track.append(mido.Message('program_change', program=program))
		# Each instrument is a track

		# Safety: sort by time
		sorted_track_list = sorted(track_list, key=lambda tup: tup[0])

		# Keep track of note on to know which one to shut down
		notes_previous = set()
		time_previous = 0
		repeated_note_previous = 0

		# print("# " + instrument_name)

		# Write events in the midi file
		for time_now, notes in sorted_track_list:

			notes_played = set()

			time = int( (time_now - time_previous) * (ticks_per_beat/quantization) ) - repeated_note_previous

			message_written = False
			repeated_note_previous = 0

			for note in notes:
				pc, octave, velocity, _, repeat = note
				
				pitch = from_pc_octave_to_pitch(pc, octave)
				
				if (pitch in notes_previous):
					# Note repeat or sustained ? If sustained do nothing
					if repeat == 1:
						track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
						# print(mido.Message('note_off', note=pitch, velocity=0, time=time))
						time = 1
						track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
						# print(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
						time = 0
						message_written = True
						repeated_note_previous += 1
					notes_played.add(pitch)
				else:
					track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
					# print(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
					notes_played.add(pitch)
					time = 0
					message_written = True
			
			# Clean removed note
			notes_to_remove = notes_previous - notes_played
			for pitch in notes_to_remove:
				track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
				# print(mido.Message('note_off', note=pitch, velocity=0, time=time))
				notes_previous.remove(pitch)
				time = 0
				message_written = True
			notes_previous = notes_previous | notes_played

			# Increment time
			if message_written:
				time_previous = time_now

		# Finish with a not off
		last_time = int(4*ticks_per_beat)
		track.append(mido.MetaMessage('end_of_track', time=last_time))


	mid.save(write_path)
	return