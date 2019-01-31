#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import re
import csv

from LOP_s2s.Database.simplify_instrumentation import get_instru_mapping, simplify_instrumentation
from LOP_s2s.Database.read_midi_seq import read_midi_seq

def get_score_from_folder(folder_path, quantization):
	instru_mapping, _ = get_instru_mapping()

	# Get midi files
	midi_files = glob.glob(folder_path + '/*.mid')
	assert (len(midi_files) == 2), ("More or less than 2 files in " + folder_path)
	csv_files = [re.sub('.mid', '.csv', e) for e in midi_files]

	with open(csv_files[0], 'r') as f0, open(csv_files[1], 'r') as f1:
		r0 = csv.DictReader(f0, delimiter=';')
		instru0 = next(r0)
		r1 = csv.DictReader(f1, delimiter=';')
		instru1 = next(r1)
	
	if len(set(instru0.values())) > len(set(instru1.values())):
		piano_midi = midi_files[1]
		piano_instru = instru1
		orch_midi = midi_files[0]
		orch_instru = instru0
	else:
		piano_midi = midi_files[0]
		piano_instru = instru0
		orch_midi = midi_files[1]
		orch_instru = instru1
	
	piano = process_file(piano_midi, piano_instru, quantization, instru_mapping=None)
	orch = process_file(orch_midi, orch_instru, quantization, instru_mapping)
	
	return piano, orch

def process_file(midi_path, instru, quantization, instru_mapping):
	# Simplify names : keep only tracks not marked as useless
	instru_simple = {k: simplify_instrumentation(v) for k, v in instru.items()}
	midi_seq = read_midi_seq(midi_path, quantization, instru_simple, instru_mapping)
	return midi_seq