#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import os
import glob
import LOP.Scripts.config as config

def avoid_tracks():

	training_avoid = [
		"bouliane/13",
		"bouliane/2",
		"bouliane/24",
		"bouliane/29",
		"bouliane/30",
		"bouliane/31",
		"bouliane/5",
		"bouliane/7",
		"hand_picked_Spotify/20",
		"hand_picked_Spotify/22",
		"hand_picked_Spotify/23",
		"hand_picked_Spotify/24",
		"hand_picked_Spotify/4", # Too different structures
		# Because of instrumentation
		"bouliane/24",
		#
		"hand_picked_Spotify/11",
		"hand_picked_Spotify/17",
		"hand_picked_Spotify/18",
		"hand_picked_Spotify/19",
		"hand_picked_Spotify/24"
		"hand_picked_Spotify/31",
		"hand_picked_Spotify/34",
		"hand_picked_Spotify/35",
		"hand_picked_Spotify/36",
		"hand_picked_Spotify/37",
		"hand_picked_Spotify/39",
		"hand_picked_Spotify/43",
		"hand_picked_Spotify/46",
		"hand_picked_Spotify/47",
		"hand_picked_Spotify/48",
		"hand_picked_Spotify/49",
		"hand_picked_Spotify/50",
		"hand_picked_Spotify/51",
		"hand_picked_Spotify/53",
		"hand_picked_Spotify/54",
		"hand_picked_Spotify/56",
		"hand_picked_Spotify/57",
		"hand_picked_Spotify/60",
		"hand_picked_Spotify/60",
		"hand_picked_Spotify/63",
		"hand_picked_Spotify/8",
		"hand_picked_Spotify/9",
		#
		"imslp/1",
		"imslp/11",
		"imslp/12",
		"imslp/13",
		"imslp/14",
		"imslp/16",
		"imslp/2",
		"imslp/20",
		"imslp/21",
		"imslp/22",
		"imslp/25",
		"imslp/26",
		"imslp/27",
		"imslp/41",
		"imslp/42",
		"imslp/48",
		"imslp/55",
		"imslp/57",
		"imslp/59",
		"imslp/61",
		"imslp/64",
		"imslp/65",
		"imslp/66",
		"imslp/7",
		"imslp/73",
		"imslp/74",
		"imslp/77",
		"imslp/78",
		"imslp/79",
		"imslp/80",
		"imslp/83",
		# Beethv 9-4 (choirs)
		"liszt_classical_archives/27"
	]

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
		# "hand_picked_Spotify/40",
		# "hand_picked_Spotify/45",
		# "imslp/21",
		# "imslp/43",
		# "imslp/20",
		# "imslp/44",
		# "imslp/22",
		# "imslp/12",
		# "imslp/14",
		# "imslp/62",
		# "imslp/68",
		# "imslp/39",
		# "imslp/15",
		# "imslp/26",
		# "imslp/71",
		# "imslp/3",
		# "imslp/78",
		# "imslp/11",
		# "imslp/86",
		# "imslp/16",
		# "imslp/25",
		# "imslp/56",
		# "imslp/77",
		# "imslp/5",
		# "imslp/23",
		# "imslp/45",
		# "imslp/50",
		# "imslp/64",
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