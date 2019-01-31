#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import time
import shutil
import os
import pickle as pkl

from LOP.Scripts.submit_job import submit_job
from LOP.Database.load_data import build_one_fold
from LOP.Scripts.training_strategies.training_strategy import Training_strategy


class TS_trAB_teA(Training_strategy):
	
	def __init__(self, num_k_folds=10, config_folder=None, database_path=None, logger=None):
		"""Train, validate and test only on A"""
		super().__init__(num_k_folds, config_folder, database_path, logger)
		return

	@staticmethod
	def name():
		return "TS_trAB_teA"

	def __build_folds(self, total_number_folds, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, embedded_piano_bool, mask_orch_bool):
		# Load files lists
		t_dict_A = pkl.load(open(self.database_path + '/train_only_A.pkl', 'rb'))
		tvt_dict_A = pkl.load(open(self.database_path + '/train_and_valid_A.pkl', 'rb'))
		t_dict_B = pkl.load(open(self.database_path + '/train_only_B.pkl', 'rb'))
		tvt_dict_B = pkl.load(open(self.database_path + '/train_and_valid_B.pkl', 'rb'))

		t_dict = {} 				# Init to avoid copy by reference
		t_dict.update(t_dict_A)
		t_dict.update(t_dict_B)
		tv_dict = {}
		tv_dict.update(tvt_dict_B) 	# Don't use B for testing
		tvt_dict = {}
		tvt_dict.update(tvt_dict_A)

		# Lists of files
		t_names_shuffle = self.sort_and_shuffle_dict(t_dict)
		tv_names_shuffle = self.sort_and_shuffle_dict(tv_dict)
		tvt_names_shuffle = self.sort_and_shuffle_dict(tvt_dict)

		if total_number_folds == -1:
			total_number_folds = len(tvt_dict)

		folds = []
		train_names = []
		valid_names = []
		test_names = []

		# Build the list of split_matrices
		for current_fold_number in range(total_number_folds):
			one_fold, this_train_names, this_valid_names, this_test_names = build_one_fold(current_fold_number, total_number_folds, 
				t_dict, tv_dict, tvt_dict, 
				t_names_shuffle, tv_names_shuffle, tvt_names_shuffle,
				temporal_order, train_batch_size, long_range_pred, 
				num_max_contiguous_blocks, embedded_piano_bool, mask_orch_bool, self.random_inst)
			folds.append(one_fold)
			train_names.append(this_train_names)
			valid_names.append(this_valid_names)
			test_names.append(this_test_names)

		return folds, train_names, valid_names, test_names

	def get_folds(self, parameters, model_params):
		# Sadly, with the split of the database, shuffling the files can only be performed inside a same split
		self.logger.info('##### Building folds')
		# Load data and build K_folds
		time_load_0 = time.time()
		# K_folds[fold_index]['train','test' or 'valid'][index split]['batches' : [[234,14,54..],[..],[..]], 'matrices_path':[path_0,path_1,..]]
		if self.num_k_folds == 0:
			# this_K_folds, this_valid_names, this_test_names = build_folds(tracks_start_end, piano, orch, 10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], RANDOM_SEED_FOLDS, logger_load=None)
			self.K_folds, self.train_names, self.valid_names, self.test_names = self.__build_folds(10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"],
				parameters["num_max_contiguous_blocks"], parameters["embedded_piano"], parameters["mask_orch"])
			self.K_folds = [self.K_folds[0]]
			self.train_names = [self.train_names[0]]
			self.valid_names = [self.valid_names[0]]
			self.test_names = [self.test_names[0]]
		elif self.num_k_folds == -1:
			self.K_folds, self.train_names, self.valid_names, self.test_names = self.__build_folds(self.num_k_folds, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"],
				parameters["num_max_contiguous_blocks"], parameters["embedded_piano"], parameters["mask_orch"])
		else:
			self.K_folds, self.train_names, self.valid_names, self.test_names = self.__build_folds(self.num_k_folds, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"],
				parameters["num_max_contiguous_blocks"], parameters["embedded_piano"], parameters["mask_orch"])
		time_load = time.time() - time_load_0
		self.logger.info('TTT : Building folds took {} seconds'.format(time_load))
		return

	def submit_jobs(self, parameters, model_params, dimensions, save_bool, generate_bool, local):
		for K_fold_ind, K_fold in enumerate(self.K_folds):
				
			# Create fold folder
			config_folder_fold = self.config_folder + "/" + str(K_fold_ind)
			if os.path.isdir(config_folder_fold):
				shutil.rmtree(config_folder_fold)
			os.mkdir(config_folder_fold)
			# Submit worker
			submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, 
				self.train_names[K_fold_ind], self.valid_names[K_fold_ind], self.test_names[K_fold_ind],
				save_bool, generate_bool, local, self.logger)
		return