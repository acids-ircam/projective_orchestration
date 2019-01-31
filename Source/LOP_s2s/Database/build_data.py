#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

# Build data for lop in seq2seq mode
# - Read midi files
# - Align (N-W)
# - Chunk
# - Store

import os
import re
import shutil
import time

from LOP_s2s import config
import LOP_s2s.Database.avoid_tracks as avoid_tracks
from LOP_s2s.Database.get_score_from_folder import get_score_from_folder
from LOP_s2s.Database.nwalign import nwalign

def build_split_matrices(folder_paths, destination_folder, chunk_size, max_number_note_played):
	file_counter = 0
	train_only_files={}
	train_and_valid_files={}

	for folder_path in folder_paths:
		###############################
		# Read file
		folder_path = folder_path.rstrip()
		print(" : " + folder_path)
		if not os.path.isdir(folder_path):
			continue

		# Avoid ?
		folder_path_split = re.split("/", folder_path)
		folder_path_relative = folder_path_split[-2] + "/" + folder_path_split[-1]
		if folder_path_relative in avoid_tracks.no_valid_tracks():
			train_only_files[folder_path_relative] = []
		else:
			train_and_valid_files[folder_path_relative] = []

		# Read midi files
		folder_path = "/Users/leo/Recherche/databases/Orchestration/LOP_database_06_09_17/bouliane/33"
		piano, orch = get_score_from_folder(folder_path, quantization)

		# Align using N-W
		import pdb; pdb.set_trace()
		(piano_align, orch_align), (piano_events, orch_events) = nwalign(piano[:100], orch[:100], gapOpen=-3, gapExtend=-1)

		# TEST WRITE FUNCTION BACK TO MIDI TO SEE IF ALIGNMENT GOES WELL
		

		# # Skip shitty files
		# if new_pr_piano is None:
		# 	# It's definitely not a match...
		# 	# Check for the files : are they really a piano score and its orchestration ??
		# 	with(open('log_build_db.txt', 'a')) as f:
		# 		f.write(folder_path + '\n')
		# 	continue

		# pr_orch = build_data_aux.cast_small_pr_into_big_pr(new_pr_orchestra, new_instru_orchestra, 0, duration, instru_mapping, np.zeros((duration, N_orchestra)))
		# pr_piano = build_data_aux.cast_small_pr_into_big_pr(new_pr_piano, {}, 0, duration, instru_mapping, np.zeros((duration, N_piano)))
		# ###############################

		# ##############################
		# # Update the max number of notes played in the orchestral score
		# this_max_num_notes = int(np.max(np.sum(pr_orch, axis=1)))
		# max_number_note_played = max(max_number_note_played, this_max_num_notes)
 	# 	##############################

		# ###############################
		# # Split
		# last_index = pr_piano.shape[0]
		# start_indices = range(0, pr_piano.shape[0], chunk_size)

		# for split_counter, start_index in enumerate(start_indices):
		# 	this_split_folder = destination_folder + '/' + str(file_counter) + '_' + str(split_counter)
		# 	os.mkdir(this_split_folder)
		# 	end_index = min(start_index + chunk_size, last_index)
		
		# 	section = pr_piano[start_index: end_index]
		# 	section_cast = section.astype(np.float32)
		# 	np.save(this_split_folder + '/pr_piano.npy', section_cast)

		# 	if build_embedding:
		# 		section = piano_embedded[start_index: end_index]
		# 		section_cast = section.astype(np.float32)
		# 		np.save(this_split_folder + '/pr_piano_embedded.npy', section_cast)

		# 	section = pr_orch[start_index: end_index]
		# 	section_cast = section.astype(np.float32)
		# 	np.save(this_split_folder + '/pr_orch.npy', section_cast)

		# 	section = new_duration_piano[start_index: end_index]
		# 	section_cast = np.asarray(section, dtype=np.int8)
		# 	np.save(this_split_folder + '/duration_piano.npy', section_cast)

		# 	section = new_duration_orch[start_index: end_index]
		# 	section_cast = np.asarray(section, dtype=np.int8)
		# 	np.save(this_split_folder + '/duration_orch.npy', section_cast)

		# 	# Keep track of splits
		# 	split_path = re.split("/", this_split_folder)
		# 	this_split_folder_relative = split_path[-2] + "/" + split_path[-1]
		# 	if folder_path_relative in avoid_tracks.no_valid_tracks():
		# 		train_only_files[folder_path_relative].append(this_split_folder_relative)
		# 	else:
		# 		train_and_valid_files[folder_path_relative].append(this_split_folder_relative)

		# file_counter+=1
		# ###############################

	# return train_and_valid_files, train_only_files, max_number_note_played
	return

def build_data(subset_A_paths, subset_B_paths, subset_C_paths, quantization, store_folder):
	
	print("##########")
	print("Build data")

	###############################
	# Iinit folders
	chunk_size = config.build_parameters()["chunk_size"]
	split_folder_A = os.path.join(store_folder, "A")
	os.mkdir(split_folder_A)
	split_folder_B = os.path.join(store_folder, "B")
	os.mkdir(split_folder_B)
	split_folder_C = os.path.join(store_folder, "C")
	os.mkdir(split_folder_C)
	
	###############################
	# Build matrices
	max_number_note_played = 0
	train_and_valid_A, train_only_A, max_number_note_played = build_split_matrices(subset_A_paths, split_folder_A, chunk_size, max_number_note_played)
	train_and_valid_B, train_only_B, max_number_note_played = build_split_matrices(subset_B_paths, split_folder_B, chunk_size, max_number_note_played)
	train_and_valid_C, train_only_C, max_number_note_played = build_split_matrices(subset_C_paths, split_folder_C, chunk_size, max_number_note_played)
	###############################

	###############################
	# Save files' lists
	pkl.dump(train_and_valid_A, open(store_folder + '/train_and_valid_A.pkl', 'wb'))
	pkl.dump(train_only_A, open(store_folder + '/train_only_A.pkl', 'wb'))
	
	pkl.dump(train_and_valid_B, open(store_folder + '/train_and_valid_B.pkl', 'wb'))
	pkl.dump(train_only_B, open(store_folder + '/train_only_B.pkl', 'wb'))
	
	pkl.dump(train_and_valid_C, open(store_folder + '/train_and_valid_C.pkl', 'wb'))
	pkl.dump(train_only_C, open(store_folder + '/train_only_C.pkl', 'wb'))
	###############################

	###############################
	# Metadata
	metadata = {}
	metadata['quantization'] = quantization
	metadata['N_orchestra'] = N_orchestra
	metadata['N_piano'] = N_piano
	metadata['N_piano_embedded'] = N_piano_embedded
	metadata['chunk_size'] = chunk_size
	metadata['instru_mapping'] = instru_mapping
	metadata['quantization'] = quantization
	metadata['temporal_granularity'] = temporal_granularity
	metadata['store_folder'] = store_folder
	metadata['n_instru'] = len(instru_mapping.keys())
	metadata['max_number_note_played'] = max_number_note_played
	if build_embedding:
		metadata['embedding_path'] = embedding_path
	else:
		metadata['embedding_path'] = ""
	with open(store_folder + '/metadata.pkl', 'wb') as outfile:
		pkl.dump(metadata, outfile)
	###############################
	return

if __name__ == '__main__':
	quantization=32
	# Database have to be built jointly so that the ranges match
	database_orchestration = config.database_root()
	database_arrangement = config.database_pretraining_root()
	
	##############################
	# Subsets
	subset_A = [
		database_orchestration + "/liszt_classical_archives", 
	]

	subset_B = [
		database_orchestration + "/bouliane", 
		database_orchestration + "/hand_picked_Spotify", 
		database_orchestration + "/imslp"
	]
	
	subset_C = [
		# database_arrangement + "/OpenMusicScores",
		# database_arrangement + "/Kunstderfuge", 
		# database_arrangement + "/Musicalion", 
		# database_arrangement + "/Mutopia"
	]
	##############################

	data_folder = config.data_root() + '/' + str(quantization)

	# Erase previous database
	if os.path.isdir(data_folder):
		shutil.rmtree(data_folder)
	os.makedirs(data_folder)

	# Create a list of paths
	def build_filepaths_list(path):
		folder_paths = []
		for file_name in os.listdir(path):
			if file_name != '.DS_Store':
				this_path = os.path.join(path, file_name)
				folder_paths.append(this_path)
		return folder_paths

	subset_A_paths = []
	for path in subset_A:
		subset_A_paths += build_filepaths_list(path)

	subset_B_paths = []
	for path in subset_B:
		subset_B_paths += build_filepaths_list(path)

	subset_C_paths = []
	for path in subset_C:
		subset_C_paths += build_filepaths_list(path)

	# Remove garbage tracks
	avoid_tracks_list = avoid_tracks.avoid_tracks()
	subset_A_paths = [e for e in subset_A_paths if e not in avoid_tracks_list]
	subset_B_paths = [e for e in subset_B_paths if e not in avoid_tracks_list]
	subset_C_paths = [e for e in subset_C_paths if e not in avoid_tracks_list]

	print("Subset A : " + str(len(subset_A_paths)))
	print("Subset B : " + str(len(subset_B_paths)))
	print("Subset C : " + str(len(subset_C_paths)))

	build_data(subset_A_paths=subset_A_paths,
			   subset_B_paths=subset_B_paths,
			   subset_C_paths=subset_C_paths,
			   quantization=quantization,
			   store_folder=data_folder)