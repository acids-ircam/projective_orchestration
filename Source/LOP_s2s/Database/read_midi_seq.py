#!/usr/bin/env python
# -*- coding: utf8 -*-

# Read midi files into a list of events:
# [
#    [(time, duration, notes)]
# ]
# 
# Where notes = [(pc(12), octave(11), intensity(128), instru_integer, repeat_flag(1))]
# each value being a integer (range in parenthesis), except instru which is a string
#
# Quantization is not made directly when reading events, so that if two very close events happen on the same pitch during the same quantized event,
# if one event was slightly before it will be dismissed (the last one only will be kept). 
# This for notes repeat cases with a very short silence between the two events

from mido import MidiFile
from unidecode import unidecode
from simplify_instrumentation import get_instru_mapping, simplify_instrumentation
import csv
import re
import numpy as np

def pitch_to_pc(pitch):
    # Pitch to pitch-class
    pc = pitch % 12 
    octave = int(pitch / 12)    # Floor it
    return pc, octave

def pitch_octave_instru(elem):
    return elem[0], elem[1], elem[3]

def search_pitch_in_block(new_elem, block):
    for elem in block:
        if pitch_octave_instru(new_elem) == pitch_octave_instru(elem):
            block.remove(elem)
            return

def midi_to_sorted_list(song_path, trackName_to_instru, instru_mapping):
    mid = MidiFile(song_path)
    # Tick per beat
    ticks_per_beat = mid.ticks_per_beat
    score = []
    possible_track_names = list(trackName_to_instru.keys())
    metadata = {}
    for track in mid.tracks:
        # Get track name
        track_name = track.name

        # Instanciate the pianoroll
        beat_counter = 0

        if track_name in possible_track_names:
            # Get instru_integer
            instru_names = trackName_to_instru[track_name]
            if "Remove" in instru_names:
                continue

            if instru_names==["Piano"]:
                # Piano
                instru_integers = [-1]
            else:
                instru_integers = [instru_mapping[e] for e in instru_names]

        else:
            for message in track:
                if message.type == 'time_signature':
                    metadata['time_signature'] = (message.numerator, message.denominator, message.clocks_per_click, message.notated_32nd_notes_per_beat)
                elif message.type == 'key_signature':
                    metadata['key_signature'] = message.key
            continue

        for message in track:
            # Time. Must be incremented, whether it is a note on/off or not
            time = float(message.time)
            beat_counter += time / ticks_per_beat
            if message.type in ['note_on', 'note_off']:
                pc, octave = pitch_to_pc(message.note)
                velocity = message.velocity
                for instru_integer in instru_integers:
                    score.append((beat_counter, (pc, octave, velocity, instru_integer)))

    # Sorted by beat
    sorted_score = sorted(score, key=lambda tup: tup[0])
    return sorted_score, metadata

def aggregate_sorted_score(sorted_score, quantization):
    # Aggregate
    score_agg = []
    time_block = 0
    block = set()
    previous_block = set()

    for elem in sorted_score:

        beat_elem = elem[0]
        new_time = int(round(beat_elem*quantization))

        # Elements are gathered by blocks
        # Copy when the new element's time is larger
        if new_time > time_block:
            # Copy previous block for:
            #   - copy by value the block written in the score
            #   - keep track of previously one notes to set repeat flag
            previous_block = set(block)
            score_agg.append((time_block, previous_block))
            time_block = new_time
            # Reset the repeat flag of all the notes in block
            block = set()
            for previous_block_tuple in previous_block:
                tuple_no_repeat = (previous_block_tuple[0], previous_block_tuple[1], previous_block_tuple[2], previous_block_tuple[3], 0)
                block.add(tuple_no_repeat) 

        new_data = elem[1]
        new_velocity = new_data[2]

        # Note on
        if new_velocity > 0:
            # If no repeat
            new_elem = new_data + (0,)
            # If in previous block: repeat
            for old_data in previous_block:
                if pitch_octave_instru(new_data)==pitch_octave_instru(old_data):
                    # Perhaps still in block (not removed previously, but that would be a bug)
                    if old_data in block:
                        block.remove(old_data)
                    # Set the repeat flag
                    new_elem = new_data + (1,)
                    break
            
            # Sometimes, because of the quantization, a note is activated twice in the same block
            # First check that the note is not already in the block
            search_pitch_in_block(new_elem, block)
            block.add(new_elem)

        # Note off
        else:
            # Remove from current block
            elem_to_remove=None
            for data_to_remove in block:
                if pitch_octave_instru(new_data)==pitch_octave_instru(data_to_remove):
                    # Possible bug, more than one note off... so need to be protected by a if
                    elem_to_remove = data_to_remove
                    break
            if elem_to_remove:
                block.remove(elem_to_remove)

    return score_agg

def read_midi_seq(song_path, quantization, trackName_to_instru, instru_mapping):
    # song_path: absolute path to midi file
    # quantization: define a minimum time interval under which events are considered the same. Symbolic value in beat division.
    # Return a list
    # trackName_to_instru: csv file mapping track names to instrument name
    # instru_mapping: mapping between instrument names and numbers
    sorted_score, metadata = midi_to_sorted_list(song_path, trackName_to_instru, instru_mapping)
    score = aggregate_sorted_score(sorted_score, quantization)
    return score, metadata


if __name__ == '__main__':
    # midi_path = '/Users/leo/Recherche/lop/database/Orchestration/LOP_database_06_09_17/liszt_classical_archives/33/beets3m1_orch.mid'
    # csv_path = '/Users/leo/Recherche/lop/database/Orchestration/LOP_database_06_09_17/liszt_classical_archives/33/beets3m1_orch.csv'

    midi_path = "/Users/leo/Recherche/lop/database/Orchestration/LOP_database_06_09_17/bouliane/0/Brahms_Symph4_iv(1-33)_ORCH+REDUC+piano_orch.mid"
    csv_path = "/Users/leo/Recherche/lop/database/Orchestration/LOP_database_06_09_17/bouliane/0/Brahms_Symph4_iv(1-33)_ORCH+REDUC+piano_orch.csv"
    
    # midi_path = "/Users/leo/Recherche/lop/database/Orchestration/LOP_database_06_09_17/bouliane/1/Schumann_AlbumJung_1(20 mesx4)_OrcBouliane_WW+piano_solo.mid"
    # csv_path = "/Users/leo/Recherche/lop/database/Orchestration/LOP_database_06_09_17/bouliane/1/Schumann_AlbumJung_1(20 mesx4)_OrcBouliane_WW+piano_solo.csv"
    
    quantization = 32

    with open(csv_path, 'r') as ff:
        rr = csv.DictReader(ff, delimiter=';')
        instru = next(rr)
    # Simplify names : keep only tracks not marked as useless
    instru_simple = {k: simplify_instrumentation(v) for k, v in instru.items()}
    instru_mapping, _ = get_instru_mapping()

    import os
    if os.path.isfile("test.mid"):
        os.remove("test.mid")
    score, metadata = read_midi_seq(midi_path, quantization, instru_simple, instru_mapping)
    from write_midi_seq import write_midi_seq
    write_midi_seq(score, quantization, 120, "test.mid", tempo=80, metadata=metadata)