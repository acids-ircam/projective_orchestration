#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import os
import glob
import LOP.Scripts.config as config

def avoid_tracks():

	training_avoid = [] 

	pre_training_avoid = [
		"Musicalion/1576",
		"Musicalion/3362",
		"Musicalion/3372",
		"Musicalion/3380",
		"Musicalion/3382",
		"Musicalion/3386",
		"Kunstderfuge/1434",
	]
	
	# return training_avoid + pre_training_avoid + tracks_with_too_few_instruments
	return training_avoid + pre_training_avoid

def no_valid_tracks():
	no_valid_tracks = [
		# Too good
		"hand_picked_Spotify/40",
		"hand_picked_Spotify/45",
		"imslp/21",
		"imslp/43",
		"imslp/20",
		"imslp/44",
		"imslp/22",
		"imslp/12",
		"imslp/14",
		"imslp/62",
		"imslp/68",
		"imslp/39",
		"imslp/15",
		"imslp/26",
		"imslp/71",
		"imslp/3",
		"imslp/78",
		"imslp/11",
		"imslp/86",
		"imslp/16",
		"imslp/25",
		"imslp/56",
		"imslp/77",
		"imslp/5",
		"imslp/23",
		"imslp/45",
		"imslp/50",
		"imslp/64",
		"debug/1",
		"debug/2",
	] 

	# All IMSLP files
	# imslp_files = glob.glob(config.database_root() + '/imslp/[0-9]*')
	# training_avoid += imslp_files

	tracks_with_too_few_instruments = []
	# with open(config.data_root() + "/few_instrument_files_pretraining.txt", 'rb') as ff:
	# 	for line in ff:
	# 		tracks_with_too_few_instruments.append(os.path.join(config.database_pretraining_root(), line.rstrip("\n")))
	with open(config.data_root() + "/few_instrument_files.txt", 'r') as ff:
		for line in ff:
			tracks_with_too_few_instruments.append(line.rstrip("\n"))
	
	return no_valid_tracks + tracks_with_too_few_instruments

if __name__ == "__main__":
	ret = avoid_tracks()