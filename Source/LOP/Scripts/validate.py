#!/usr/bin/env python
# -*- coding: utf8 -*-

import time
import os
from multiprocessing.pool import ThreadPool
import numpy as np
import shutil

import config
from LOP.Scripts.asynchronous_load_mat import async_load_mat
from LOP.Utils.measure import accuracy_measure, precision_measure, recall_measure, true_accuracy_measure, f_measure, binary_cross_entropy

def validate(trainer, sess, init_matrices_validation, valid_splits_batches, normalizer, parameters, logger):

	temporal_order = trainer.temporal_order

	if trainer.DEBUG["save_measures"]:
		preds = []
		truth = []
	else:
		preds = None
		truth = None

	accuracy = []
	precision = []
	recall = []
	val_loss = []
	true_accuracy = []
	f_score = []
	Xent = []

	accuracy_sampled = []
	precision_sampled = []
	recall_sampled = []
	val_loss_sampled = []
	true_accuracy_sampled = []
	f_score_sampled = []
	Xent_sampled = []

	accuracy_long_range = []
	precision_long_range = []
	recall_long_range = []
	val_loss_long_range = []
	true_accuracy_long_range = []
	f_score_long_range = []
	Xent_long_range = []

	N_matrix_files = len(valid_splits_batches)
	pool = ThreadPool(processes=1)
	matrices_from_thread = init_matrices_validation

	for file_ind_CURRENT in range(N_matrix_files):
		#######################################
		# Get indices and matrices to load
		#######################################
		# We train on the current matrix
		valid_index = valid_splits_batches[file_ind_CURRENT]['batches']
		valid_long_range_index = valid_splits_batches[file_ind_CURRENT]['batches_lr']
		# But load the one next one
		file_ind_NEXT = (file_ind_CURRENT+1) % N_matrix_files
		next_chunks = valid_splits_batches[file_ind_NEXT]['chunks_folders']
		
		#######################################
		# Load matrix thread
		#######################################
		async_valid = pool.apply_async(async_load_mat, (normalizer, next_chunks, parameters, trainer.np_type))

		piano_input, orch_input, duration_piano, mask_orch = matrices_from_thread
			
		trainer.DEBUG["num_batch"] = len(valid_index)
		
		#######################################
		# Loop for short-term validation
		#######################################
		for batch_counter, batch_index in enumerate(valid_index):

			# if type(trainer).__name__ == 'NADE_trainer':
			# 	if batch_index > 5:
			# 		break

			trainer.DEBUG["batch_counter"] = batch_counter

			loss_batch, preds_batch, orch_t = trainer.valid_step(sess, batch_index, piano_input, orch_input, duration_piano, mask_orch)

			##############################
			Xent_batch = binary_cross_entropy(orch_t, preds_batch)
			accuracy_batch = accuracy_measure(orch_t, preds_batch)
			precision_batch = precision_measure(orch_t, preds_batch)
			recall_batch = recall_measure(orch_t, preds_batch)
			true_accuracy_batch = true_accuracy_measure(orch_t, preds_batch)
			f_score_batch = f_measure(orch_t, preds_batch)
			#
			val_loss.extend(loss_batch)
			accuracy.extend(accuracy_batch)
			precision.extend(precision_batch)
			recall.extend(recall_batch)
			true_accuracy.extend(true_accuracy_batch)
			f_score.extend(f_score_batch)
			Xent.extend(Xent_batch)
			##############################

			##############################
			preds_batch_sampled = np.random.binomial(1, preds_batch)
			Xent_batch = binary_cross_entropy(orch_t, preds_batch_sampled)
			accuracy_batch = accuracy_measure(orch_t, preds_batch_sampled)
			precision_batch = precision_measure(orch_t, preds_batch_sampled)
			recall_batch = recall_measure(orch_t, preds_batch_sampled)
			true_accuracy_batch = true_accuracy_measure(orch_t, preds_batch_sampled)
			f_score_batch = f_measure(orch_t, preds_batch_sampled)
			#
			val_loss_sampled.extend(loss_batch)
			accuracy_sampled.extend(accuracy_batch)
			precision_sampled.extend(precision_batch)
			recall_sampled.extend(recall_batch)
			true_accuracy_sampled.extend(true_accuracy_batch)
			f_score_sampled.extend(f_score_batch)
			Xent_sampled.extend(Xent_batch)
			##############################

			if trainer.DEBUG["save_measures"]:
				# No need to store all the training points
				preds.extend(preds_batch)
				truth.extend(orch_t)

		#######################################
		# Loop for long-term validation
		# The task is filling a gap of size parameters["long_range"]
		# So the algo is given :
		#       orch[0:temporal_order]
		#       orch[temporal_order+parameters["long_range"]:
		#            (2*temporal_order)+parameters["long_range"]]
		# And must fill :
		#       orch[temporal_order:
		#            temporal_order+parameters["long_range"]]
		#######################################
		for batch_index in valid_long_range_index:
			# Init
			# Extract from piano and orchestra the matrices required for the task
			seq_len = (temporal_order-1) * 2 + parameters["long_range"]
			piano_dim = piano_input.shape[1] 
			orch_dim = orch_input.shape[1]
			piano_extracted = np.zeros((len(batch_index), seq_len, piano_dim))
			orch_extracted = np.zeros((len(batch_index), seq_len, orch_dim))
			if parameters["duration_piano"]:
				duration_piano_extracted = np.zeros((len(batch_index), seq_len))
			else:
				duration_piano_extracted = None
			orch_gen = np.zeros((len(batch_index), seq_len, orch_dim))

			for ind_b, this_batch_ind in enumerate(batch_index):
				start_ind = this_batch_ind-temporal_order+1
				end_ind = start_ind + seq_len
				piano_extracted[ind_b] = piano_input[start_ind:end_ind,:]
				orch_extracted[ind_b] = orch_input[start_ind:end_ind,:]
				if parameters["duration_piano"]:
					duration_piano_extracted[ind_b] = duration_piano[start_ind:end_ind]
			
			# We know the past orchestration at the beginning...
			orch_gen[:, :temporal_order-1, :] = orch_extracted[:, :temporal_order-1, :]
			# and the future orchestration at the end
			orch_gen[:, -temporal_order+1:, :] = orch_extracted[:, -temporal_order+1:, :]
			# check we didn't gave the correct information
			assert orch_gen[:, temporal_order-1:(temporal_order-1)+parameters["long_range"], :].sum()==0, "The gap to fill in orch_gen contains values !"

			for t in range(temporal_order-1, temporal_order-1+parameters["long_range"]):

				loss_batch, preds_batch, orch_t = trainer.valid_long_range_step(sess, t, piano_extracted, orch_extracted, orch_gen, duration_piano_extracted)

				prediction_sampled = np.random.binomial(1, preds_batch)
				orch_gen[:, t, :] = prediction_sampled

				# Compute performances measures
				Xent_batch = binary_cross_entropy(orch_t, preds_batch)
				accuracy_batch = accuracy_measure(orch_t, preds_batch)
				precision_batch = precision_measure(orch_t, preds_batch)
				recall_batch = recall_measure(orch_t, preds_batch)
				true_accuracy_batch = true_accuracy_measure(orch_t, preds_batch)
				f_score_batch = f_measure(orch_t, preds_batch)

				val_loss_long_range.extend(loss_batch)
				accuracy_long_range.extend(accuracy_batch)
				precision_long_range.extend(precision_batch)
				recall_long_range.extend(recall_batch)
				true_accuracy_long_range.extend(true_accuracy_batch)
				f_score_long_range.extend(f_score_batch)
				Xent_long_range.extend(Xent_batch)

		del(matrices_from_thread)
		matrices_from_thread = async_valid.get()

	pool.close()
	pool.join()

	valid_results = {
		'accuracy': np.asarray(accuracy), 
		'precision': np.asarray(precision), 
		'recall': np.asarray(recall), 
		'loss': np.asarray(val_loss), 
		'true_accuracy': np.asarray(true_accuracy), 
		'f_score': np.asarray(f_score), 
		'Xent': np.asarray(Xent)
		}
	valid_results_sampled = {
		'accuracy': np.asarray(accuracy_sampled), 
		'precision': np.asarray(precision_sampled), 
		'recall': np.asarray(recall_sampled), 
		'loss': np.asarray(val_loss_sampled), 
		'true_accuracy': np.asarray(true_accuracy_sampled), 
		'f_score': np.asarray(f_score_sampled), 
		'Xent': np.asarray(Xent_sampled)
		}
	valid_long_range_results = {
		'accuracy': np.asarray(accuracy_long_range),
		'precision': np.asarray(precision_long_range), 
		'recall': np.asarray(recall_long_range),
		'loss': np.asarray(val_loss_long_range), 
		'true_accuracy': np.asarray(true_accuracy_long_range), 
		'f_score': np.asarray(f_score_long_range), 
		'Xent': np.asarray(Xent_long_range)
		}
	return valid_results, valid_results_sampled, valid_long_range_results, preds, truth