#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

#################################################
#################################################
#################################################
# Note :
#   - pitch range for each instrument is set based on the observed pitch range of the database
#   - for test set, we picked seminal examples. See the name_db_{test;train;valid}.txt files that
#       list the files path :
#           - beethoven/liszt symph 5 - 1 : liszt_classical_archive/16
#           - mouss/ravel pictures exhib : bouliane/22
#################################################
#################################################
#################################################

import os
import glob
import shutil
import re
import numpy as np
import LOP.Scripts.config as config
import build_data_aux
import build_data_aux_no_piano
import pickle as pkl
import avoid_tracks 
import torch
# memory issues
import gc
import sys
import time

import LOP.Scripts.config as config

import LOP_database.utils.reconstruct_pr as reconstruct_pr
from LOP_database.midi.write_midi import write_midi

DEBUG=False
ERASE=True

cuda_gpu = torch.cuda.is_available()

def update_instru_mapping(folder_path, instru_mapping, index_instru, quantization):
	if not os.path.isdir(folder_path):
		return instru_mapping, index_instru
	
	# Is there an original piano score or do we have to create it ?
	num_music_file = len(glob.glob(folder_path + '/*.xml'))
	if num_music_file == 2:
		is_piano = True
	elif num_music_file == 1:
		is_piano = False
	else:
		raise Exception("More than two or zero file(s)")

	# Read pr
	if is_piano:
		pr_piano, _, _, _, pr_orch, _, _, _, duration =\
			build_data_aux.process_folder(folder_path, quantization, binary_piano, binary_orch, temporal_granularity, align_bool=False)
	else:
		try:
			pr_piano, _, _, _, pr_orch, _, _, _, duration =\
				build_data_aux_no_piano.process_folder_NP(folder_path, quantization, binary_piano, binary_orch, temporal_granularity)
		except:
			duration=None
			logging.warning("Could not read file in " + folder_path)
	
	if duration is None:
		# Files that could not be aligned
		return instru_mapping, index_instru
	
	# Modify the mapping from instrument to indices in pianorolls and pitch bounds
	instru_mapping, index_instru = build_data_aux.instru_pitch_range(pr=pr_piano,
													   instru_mapping=instru_mapping,
													   index_instru=index_instru
													   )
	# remark : instru_mapping would be modified if it is only passed to the function,
	#                   f(a)  where a is modified inside the function
	# but i prefer to make the reallocation explicit
	#                   a = f(a) with f returning the modified value of a.
	# Does it change anything for computation speed ? (Python pass by reference,
	# but a slightly different version of it, not clear to me)
	instru_mapping, index_instru = build_data_aux.instru_pitch_range(pr=pr_orch,
													   instru_mapping=instru_mapping,
													   index_instru=index_instru
													   )

	return instru_mapping, index_instru


# def build_instru_mapping(subset_A_paths, subset_B_paths, subset_C_paths, meta_info_path, quantization, temporal_granularity, logging=None):
# 	logging.info("##########")
# 	logging.info("Get dimension informations")
# 	# Determine the temporal size of the matrices
# 	# If the two files have different sizes, we use the shortest (to limit the use of memory,
# 	# we better contract files instead of expanding them).
# 	# Get instrument names
# 	instru_mapping = {}
# 	# instru_mapping = {'piano': {'pitch_min': 24, 'pitch_max':117, 'ind_min': 0, 'ind_max': 92},
# 	#                         'harp' ... }
	
# 	folder_paths_splits_A = {}
# 	folder_paths_splits_B = {}
# 	folder_paths_splits_C = {}

# 	index_instru = 0

# 	##############################
# 	# Subset A
# 	for folder_path in subset_A_paths:
# 		folder_path = folder_path.rstrip()
# 		instru_mapping, index_instru = update_instru_mapping(folder_path, instru_mapping, index_instru, quantization)
# 	##############################

# 	##############################
# 	# Subset B
# 	for folder_path in subset_B_paths:
# 		folder_path = folder_path.rstrip()
# 		instru_mapping, index_instru = update_instru_mapping(folder_path, instru_mapping, index_instru, quantization)
# 	##############################

# 	##############################
# 	# Subset C
# 	for folder_path in subset_C_paths:
# 		folder_path = folder_path.rstrip()
# 		instru_mapping, index_instru = update_instru_mapping(folder_path, instru_mapping, index_instru, quantization)
# 	##############################

# 	##############################
# 	# Build the index_min and index_max in the instru_mapping dictionary
# 	counter = 0
# 	for k, v in instru_mapping.items():
# 		if k == 'Piano':
# 			index_min = 0
# 			index_max = v['pitch_max'] - v['pitch_min']
# 			v['index_min'] = index_min
# 			v['index_max'] = index_max
# 			continue
# 		index_min = counter
# 		counter = counter + v['pitch_max'] - v['pitch_min']
# 		index_max = counter
# 		v['index_min'] = index_min
# 		v['index_max'] = index_max
# 	##############################

# 	##############################
# 	# Save the parameters
# 	temp = {}
# 	temp['instru_mapping'] = instru_mapping
# 	temp['quantization'] = quantization
# 	temp['N_orchestra'] = counter
# 	pkl.dump(temp, open(meta_info_path, 'wb'))
# 	##############################
# 	return

def build_instru_mapping_AB(subset_A_paths, subset_B_paths, meta_info_path, quantization, temporal_granularity, logging=None):
	logging.info("##########")
	logging.info("Get dimension informations")
	# Determine the temporal size of the matrices
	# If the two files have different sizes, we use the shortest (to limit the use of memory,
	# we better contract files instead of expanding them).
	# Get instrument names
	instru_mapping = {}
	# instru_mapping = {'piano': {'pitch_min': 24, 'pitch_max':117, 'ind_min': 0, 'ind_max': 92},
	#                         'harp' ... }
	index_instru = 0

	##############################
	# Subset A
	for folder_path in subset_A_paths:
		folder_path = folder_path.rstrip()
		instru_mapping, index_instru = update_instru_mapping(folder_path, instru_mapping, index_instru, quantization)
	##############################

	##############################
	# Subset B
	for folder_path in subset_B_paths:
		folder_path = folder_path.rstrip()
		instru_mapping, index_instru = update_instru_mapping(folder_path, instru_mapping, index_instru, quantization)
	##############################

	##############################
	# Build reference
	reference_pitch_ranges = {}
	for instru in ['Bassoon', 'Clarinet', 'Contrabass', 'Flute','Harp','Horn','Oboe','Timpani','Trombone','Trumpet','Tuba','Viola','Violin','Violoncello', 'Piano']:
		reference_pitch_ranges[instru] = np.zeros((128,))
	# Reference tessitura, taken from
	# http://www.orchestralibrary.com/reftables/rang.html
	# CAREFUL: sounding pitches are used, not writen one,which differ for transposing instruments
	reference_pitch_ranges['Bassoon'] = (34,75)
	reference_pitch_ranges['Clarinet'] = (52,96)
	reference_pitch_ranges['Contrabass'] = (24+12,60+12)
	reference_pitch_ranges['Flute'] = (55,91+3)
	reference_pitch_ranges['Harp'] = (23,102)
	reference_pitch_ranges['Horn'] = (35,77)
	reference_pitch_ranges['Oboe'] = (58,93)
	reference_pitch_ranges['Timpani'] = (38,60)
	reference_pitch_ranges['Trombone'] = (40,77)
	reference_pitch_ranges['Trumpet'] = (54,86)
	reference_pitch_ranges['Tuba'] = (26,65)
	reference_pitch_ranges['Viola'] = (48,88)
	reference_pitch_ranges['Violin'] = (55,105)
	reference_pitch_ranges['Violoncello'] = (36,84)
	#
	reference_pitch_ranges['Piano'] = (21,108)
	##############################

	##############################
	# Modify Pitch min and max according to reference pitches
	instru_mapping_ref = {}
	for k, v in instru_mapping.items():
		instru_mapping_ref[k] = {}
		instru_mapping_ref[k]['pitch_min'] = max(v['pitch_min'], reference_pitch_ranges[k][0])
		instru_mapping_ref[k]['pitch_max'] = min(v['pitch_max'], reference_pitch_ranges[k][1])
	##############################

	##############################
	# Build the index_min and index_max in the instru_mapping dictionary
	counter = 0
	for k, v in instru_mapping_ref.items():
		if k == 'Piano':
			index_min = 0
			index_max = v['pitch_max'] - v['pitch_min']
			v['index_min'] = index_min
			v['index_max'] = index_max
			continue
		index_min = counter
		counter = counter + v['pitch_max'] - v['pitch_min']
		index_max = counter
		v['index_min'] = index_min
		v['index_max'] = index_max
	##############################

	##############################
	# Save the parameters
	temp = {}
	temp['instru_mapping'] = instru_mapping_ref
	temp['quantization'] = quantization
	temp['N_orchestra'] = counter
	pkl.dump(temp, open(meta_info_path, 'wb'))
	##############################
	return

def build_split_matrices(folder_paths, destination_folder, chunk_size, instru_mapping, N_piano, N_orchestra, embedding_model, binary_piano, binary_orch, build_embedding, max_number_note_played):
	file_counter = 0
	train_only_files={}
	train_and_valid_files={}

	for folder_path in folder_paths:
		###############################
		# Read file
		folder_path = folder_path.rstrip()
		logging.info(" : " + folder_path)
		if not os.path.isdir(folder_path):
			continue

		folder_path_split = re.split("/", folder_path)
		folder_path_relative = folder_path_split[-2] + "/" + folder_path_split[-1]
		if folder_path_relative in avoid_tracks.no_valid_tracks():
			train_only_files[folder_path_relative] = []
		else:
			train_and_valid_files[folder_path_relative] = []

		# Is there an original piano score or do we have to create it ?
		num_music_file = max(len(glob.glob(folder_path + '/*.mid')), len(glob.glob(folder_path + '/*.xml')))
		if num_music_file == 2:
			is_piano = True
		elif num_music_file == 1:
			is_piano = False
		else:
			raise Exception("CAVAVAVAMAVAL")

		# Get pr, warped and duration
		if is_piano:
			new_pr_piano, _, new_duration_piano, new_name_piano, new_pr_orchestra, _, new_duration_orch, new_name_orchestra, duration\
				= build_data_aux.process_folder(folder_path, quantization, binary_piano, binary_orch, temporal_granularity, gapopen=3, gapextend=1)
		else:
			try:
				new_pr_piano, _, new_duration_piano, _, new_name_piano, new_pr_orchestra, _, new_duration_orch, new_name_orchestra, duration\
					= build_data_aux_no_piano.process_folder_NP(folder_path, quantization, binary_piano, binary_orch, temporal_granularity)
			except:
				logging.warning("Could not read file in " + folder_path)
				continue

		# Skip shitty files
		if new_pr_piano is None:
			# It's definitely not a match...
			# Check for the files : are they really a piano score and its orchestration ??
			with(open('log_build_db.txt', 'a')) as f:
				f.write(folder_path + '\n')
			continue

		pr_orch = build_data_aux.cast_small_pr_into_big_pr(new_pr_orchestra, 0, duration, instru_mapping, np.zeros((duration, N_orchestra)))
		pr_piano = build_data_aux.cast_small_pr_into_big_pr(new_pr_piano, 0, duration, instru_mapping, np.zeros((duration, N_piano)))
		
		##############################
		# Bonus: write aligned midi files in an external folder 
		# Reconstruct rhythm
		# pr_orchestra_I = reconstruct_pr.instrument_reconstruction(pr_orch, instru_mapping)
		# pr_piano_I = reconstruct_pr.instrument_reconstruction_piano(pr_piano, instru_mapping)
		# # Write midi
		# target_folder = re.sub("LOP_database_mxml_clean", "LOP_database_event_aligned", folder_path)
		# os.makedirs(target_folder)
		# target_piano = re.sub("LOP_database_mxml_clean", "LOP_database_event_aligned", new_name_piano) + '.mid'
		# target_orchestra = re.sub("LOP_database_mxml_clean", "LOP_database_event_aligned", new_name_orchestra) + '.mid'
		# write_midi({k: v*90 for k,v in pr_piano_I.items()}, ticks_per_beat=1, write_path=target_piano, articulation=None)
		# write_midi({k: v*90 for k,v in pr_orchestra_I.items() if (v.sum()>0)}, ticks_per_beat=1, write_path=target_orchestra, articulation=None)
		###############################
		
		###############################
		# Embed piano
		if build_embedding:
			piano_embedded = []
			len_piano = len(pr_piano)
			batch_size = 500  			# forced to batch for memory issues
			start_batch_index = 0
			while start_batch_index < len_piano:
				end_batch_index = min(start_batch_index+batch_size, len_piano)
				this_batch_size = end_batch_index-start_batch_index
				piano_resize_emb = np.zeros((this_batch_size, 1, 128)) # Embeddings accetp size 128 samples
				piano_resize_emb[:, 0, instru_mapping['Piano']['pitch_min']:instru_mapping['Piano']['pitch_max']] = pr_piano[start_batch_index:end_batch_index]
				piano_resize_emb_TT = torch.tensor(piano_resize_emb)
				if cuda_gpu:
					piano_resize_emb_TT = piano_resize_emb_TT.cuda()
				piano_embedded_TT = embedding_model(piano_resize_emb_TT.float(), 0)
				if cuda_gpu:
					piano_embedded.append(piano_embedded_TT.cpu().numpy())
				else:
					piano_embedded.append(piano_embedded_TT.numpy())
				start_batch_index+=batch_size
			piano_embedded = np.concatenate(piano_embedded)
		###############################

		##############################
		# Update the max number of notes played in the orchestral score
		this_max_num_notes = int(np.max(np.sum(pr_orch, axis=1)))
		max_number_note_played = max(max_number_note_played, this_max_num_notes)
		##############################

		###############################
		# Split
		last_index = pr_piano.shape[0]
		start_indices = range(0, pr_piano.shape[0], chunk_size)

		for split_counter, start_index in enumerate(start_indices):
			this_split_folder = destination_folder + '/' + str(file_counter) + '_' + str(split_counter)
			os.mkdir(this_split_folder)
			end_index = min(start_index + chunk_size, last_index)
		
			section = pr_piano[start_index: end_index]
			section_cast = section.astype(np.float32)
			np.save(this_split_folder + '/pr_piano.npy', section_cast)

			if build_embedding:
				section = piano_embedded[start_index: end_index]
				section_cast = section.astype(np.float32)
				np.save(this_split_folder + '/pr_piano_embedded.npy', section_cast)

			section = pr_orch[start_index: end_index]
			section_cast = section.astype(np.float32)
			np.save(this_split_folder + '/pr_orch.npy', section_cast)

			section = new_duration_piano[start_index: end_index]
			section_cast = np.asarray(section, dtype=np.int8)
			np.save(this_split_folder + '/duration_piano.npy', section_cast)

			section = new_duration_orch[start_index: end_index]
			section_cast = np.asarray(section, dtype=np.int8)
			np.save(this_split_folder + '/duration_orch.npy', section_cast)

			# Keep track of splits
			split_path = re.split("/", this_split_folder)
			this_split_folder_relative = split_path[-2] + "/" + split_path[-1]
			if folder_path_relative in avoid_tracks.no_valid_tracks():
				train_only_files[folder_path_relative].append(this_split_folder_relative)
			else:
				train_and_valid_files[folder_path_relative].append(this_split_folder_relative)

		file_counter+=1
		###############################

	return train_and_valid_files, train_only_files, max_number_note_played

def build_data(subset_A_paths, subset_B_paths, subset_C_paths, meta_info_path, quantization, temporal_granularity, binary_piano, binary_orch, build_embedding, store_folder, logging=None):
	
	# build_instru_mapping(subset_A_paths, subset_B_paths, subset_C_paths, meta_info_path=meta_info_path, quantization=quantization, temporal_granularity=temporal_granularity, logging=logging)
	build_instru_mapping_AB(subset_A_paths, subset_B_paths, meta_info_path=meta_info_path, quantization=quantization, temporal_granularity=temporal_granularity, logging=logging)

	import pdb; pdb.set_trace()

	logging.info("##########")
	logging.info("Build data")

	###############################
	# Load embedding model
	if build_embedding:
		embedding_path = config.database_embedding() + "/Model_LEO-CLEAN_multi_0_none_input_0_none_FINAL.pth"
		embedding_model = torch.load(embedding_path, map_location="cpu")
		if cuda_gpu:
			embedding_model.cuda()
		N_piano_embedded = embedding_model.LSTM.weight_ih_l0.shape[1]
	else:
		embedding_model = None
		N_piano_embedded = 0
	###############################

	###############################
	# Load temp.pkl
	temp = pkl.load(open(meta_info_path, 'rb'))
	instru_mapping = temp['instru_mapping']
	quantization = temp['quantization']
	N_orchestra = temp['N_orchestra']
	N_piano = instru_mapping['Piano']['index_max']
	###############################

	###############################
	# Build the pitch and instru indicator vectors
	# We use integer to identify pitches and instrument
	# Used for NADE rule-based masking, not for reconstruction
	pitch_orch = np.zeros((N_orchestra), dtype="int8")-1
	instru_orch = np.zeros((N_orchestra), dtype="int8")-1
	counter = 0
	for k, v in instru_mapping.items():
		if k == "Piano":
			continue
		pitch_orch[v['index_min']:v['index_max']] = np.arange(v['pitch_min'], v['pitch_max']) % 12
		instru_orch[v['index_min']:v['index_max']] = counter
		counter += 1
	pitch_piano = np.arange(instru_mapping['Piano']['pitch_min'], instru_mapping['Piano']['pitch_max'], dtype='int8') % 12
	np.save(store_folder + '/pitch_orch.npy', pitch_orch)
	np.save(store_folder + '/instru_orch.npy', instru_orch)
	np.save(store_folder + '/pitch_piano.npy', pitch_piano)
	###############################

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
	train_and_valid_A, train_only_A, max_number_note_played = build_split_matrices(subset_A_paths, split_folder_A, chunk_size, instru_mapping, N_piano, N_orchestra, embedding_model, binary_piano, binary_orch, build_embedding, max_number_note_played)
	train_and_valid_B, train_only_B, max_number_note_played = build_split_matrices(subset_B_paths, split_folder_B, chunk_size, instru_mapping, N_piano, N_orchestra, embedding_model, binary_piano, binary_orch, build_embedding, max_number_note_played)
	train_and_valid_C, train_only_C, max_number_note_played = build_split_matrices(subset_C_paths, split_folder_C, chunk_size, instru_mapping, N_piano, N_orchestra, embedding_model, binary_piano, binary_orch, build_embedding, max_number_note_played)
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
	import logging
	# log file
	log_file_path = config.scratch_space() + '/log_build_data'
	# set up logging to file - see previous section for more details
	logging.basicConfig(level=logging.INFO,
						format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
						datefmt='%m-%d %H:%M',
						filename=log_file_path,
						filemode='w')
	# define a Handler which writes INFO messages or higher to the sys.stderr
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	# set a format which is simpler for console use
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	# tell the handler to use this format
	console.setFormatter(formatter)
	# add the handler to the root logger
	logging.getLogger('').addHandler(console)

	# Set up
	# NOTE : can't do data augmentation with K-folds, or it would require to build K times the database
	# because train is data augmented but not test and validate
	temporal_granularity='event_level'
	quantization=32
	binary_piano=True
	binary_orch=True
	build_embedding=False

	# Database have to be built jointly so that the ranges match
	database_orchestration = config.database_root()
	database_arrangement = config.database_pretraining_root()
	
	##############################
	# Subsets
	subset_A = [
		database_orchestration + "/liszt_classical_archives",
		# database_orchestration + "/debug",
	]

	subset_B = [
		# database_orchestration + "/bouliane", 
		# database_orchestration + "/hand_picked_Spotify", 
		# database_orchestration + "/imslp"
	]
	
	subset_C = [
		# database_arrangement + "/OpenMusicScores",
		# database_arrangement + "/Kunstderfuge", 
		# database_arrangement + "/Musicalion", 
		# database_arrangement + "/Mutopia"
	]
	##############################

	data_folder = config.data_root() + '/Data_A_ref'
	if DEBUG:
		data_folder += '_DEBUG'
	if binary_piano:
		data_folder += '_bp'
	if binary_orch:
		data_folder += '_bo'
	if not build_embedding:
		data_folder += '_noEmb'
	data_folder += '_tempGran' + str(quantization)

	if ERASE:
		if os.path.isdir(data_folder):
			shutil.rmtree(data_folder)
		os.makedirs(data_folder)

	ff=open(data_folder + '/binary_piano', 'wb')
	ff.close()
	open(data_folder + '/binary_orch', 'wb').close()

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
			   meta_info_path=data_folder + '/temp.pkl',
			   quantization=quantization,
			   temporal_granularity=temporal_granularity,
			   binary_piano=binary_piano,
			   binary_orch=binary_orch,
			   build_embedding=build_embedding,
			   store_folder=data_folder,
			   logging=logging)