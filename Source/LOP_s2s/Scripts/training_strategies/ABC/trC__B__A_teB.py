#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import time
import random
import shutil
import os
import pickle as pkl

from LOP.Scripts.submit_job import submit_job
from LOP.Database.load_data import build_one_fold


class TS_trC__B__A_teB(object):
	
	def __init__(self, num_k_folds=10, config_folder=None, database_path=None, logger=None):
		"""Train, validate and test only on A
		"""
		self.num_k_folds = num_k_folds
		self.config_folder = config_folder
		self.database_path = database_path
		self.logger = logger
		# Important for reproducibility
		self.random_inst = random.Random()
		self.random_inst.seed(1234)
		return

	@staticmethod
	def name():
		return "TS_trC__B__A_teB"

	def __build_folds(self, total_number_folds, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, embedded_piano_bool, mask_orch_bool):
		# Load files lists
		t_dict_A = pkl.load(open(self.database_path + '/train_only_A.pkl', 'rb'))
		tv_dict_A = pkl.load(open(self.database_path + '/train_and_valid_A.pkl', 'rb'))
		t_dict_B = pkl.load(open(self.database_path + '/train_only_B.pkl', 'rb'))
		tvt_dict_B = pkl.load(open(self.database_path + '/train_and_valid_B.pkl', 'rb'))
		t_dict_C = pkl.load(open(self.database_path + '/train_only_C.pkl', 'rb'))
		tv_dict_C = pkl.load(open(self.database_path + '/train_and_valid_C.pkl', 'rb'))

		if total_number_folds == -1:
			total_number_folds = len(tvt_dict_B.keys())

		folds_step1 = []
		train_names_1 = []
		valid_names_1 = []
		test_names_1 = []
		folds_step2 = []
		train_names_2 = []
		valid_names_2 = []
		test_names_2 = []

		# Pretraining is done one time on one single fold (8/1/1 split -> number of k-folds = 10)
		fold_0, train_names_0, valid_names_0, test_names_0 = build_one_fold(0, 10, t_dict_C, tv_dict_C, {}, temporal_order, train_batch_size, long_range_pred, 
			num_max_contiguous_blocks, embedded_piano_bool, mask_orch_bool, self.random_inst)

		# Build the list of split_matrices
		for current_fold_number in range(total_number_folds):
			# Pretraining is done one time on one single fold (8/1/1 split -> number of k-folds = 10)
			one_fold_B, this_train_names_B, this_valid_names_B, this_test_names_B = build_one_fold(current_fold_number, total_number_folds, t_dict_B, {}, tvt_dict_B, temporal_order, train_batch_size, long_range_pred, 
				num_max_contiguous_blocks, embedded_piano_bool, mask_orch_bool, self.random_inst)
			one_fold_A, this_train_names_A, this_valid_names_A, this_test_names_A = build_one_fold(current_fold_number, total_number_folds, t_dict_A, tv_dict_A, {}, temporal_order, train_batch_size, long_range_pred, 
				num_max_contiguous_blocks, embedded_piano_bool, mask_orch_bool, self.random_inst)

			folds_step1.append(one_fold_B)
			train_names_1.append(this_train_names_B)
			valid_names_1.append(this_valid_names_B)
			test_names_1.append(this_test_names_B)
			
			one_fold_step2 = {
				'train': one_fold_A['train'] + one_fold_A['test'], 	# we can use test data in this case
				'valid': one_fold_A['valid'],
				'test': one_fold_B['test']
			}
			folds_step2.append(one_fold_step2)
			train_names_2.append(this_train_names_A + this_test_names_A)
			valid_names_2.append(this_valid_names_A)
			test_names_2.append(this_test_names_B)

		return fold_0, train_names_0, valid_names_0, test_names_0, folds_step1, train_names_1, valid_names_1, test_names_1, folds_step2, train_names_2, valid_names_2, test_names_2

	def get_folds(self, parameters, model_params):
		# Sadly, with the split of the database, shuffling the files can only be performed inside a same split
		self.logger.info('##### Building folds')
		# Load data and build K_folds
		time_load_0 = time.time()
		# K_folds[fold_index]['train','test' or 'valid'][index split]['batches' : [[234,14,54..],[..],[..]], 'matrices_path':[path_0,path_1,..]]
		if self.num_k_folds == 0:
			# this_K_folds, this_valid_names, this_test_names = build_folds(tracks_start_end, piano, orch, 10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], RANDOM_SEED_FOLDS, logger_load=None)
			fold_0, train_names_0, valid_names_0, test_names_0, folds_1, train_names_1, valid_names_1, test_names_1, folds_2, train_names_2, valid_names_2, test_names_2 = \
				self.__build_folds(10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], 
					parameters["num_max_contiguous_blocks"], parameters["embedded_piano"], parameters["mask_orch"])
			self.K_folds_0 = fold_0
			self.train_names_0 = train_names_0
			self.valid_names_0 = valid_names_0
			self.test_names_0 = test_names_0
			self.K_folds_1 = [folds_1[0]]
			self.train_names_1 = [train_names_1[0]]
			self.valid_names_1 = [valid_names_1[0]]
			self.test_names_1 = [test_names_1[0]]
			self.K_folds_2 = [folds_2[0]]
			self.train_names_2 = [train_names_2[0]]
			self.valid_names_2 = [valid_names_2[0]]
			self.test_names_2 = [test_names_1[0]]
		elif self.num_k_folds == -1:
			raise Exception("num_k_folds = -1 Doesn't really make sense here")
		else:
			self.K_folds_0, self.train_names_0, self.valid_names_0, self.test_names_0, self.K_folds_1, self.train_names_1, self.valid_names_1, self.test_names_1, self.K_folds_2, self.train_names_2, self.valid_names_2, self.test_names_2 \
				= self.__build_folds(self.num_k_folds, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], 
					parameters["num_max_contiguous_blocks"], parameters["embedded_piano"], parameters["mask_orch"])
		time_load = time.time() - time_load_0
		self.logger.info('TTT : Building folds took {} seconds'.format(time_load))
		return

	def submit_jobs(self, parameters, model_params, dimensions, save_bool, generate_bool, local):

		parameters['pretrained_model'] = None
		
		config_folder_0 = self.config_folder + "/step0"
		if os.path.isdir(config_folder_0):
			shutil.rmtree(config_folder_0)
		os.mkdir(config_folder_0)
		# Submit worker
		submit_job(config_folder_0, parameters, model_params, dimensions, self.K_folds_0, 
			self.train_names_0, self.valid_names_0, self.test_names_0,
			True, False, local, self.logger)
		

		for K_fold_ind, (K_fold_1, K_fold_2) in enumerate(zip(self.K_folds_1, self.K_folds_2)):

			parameters['pretrained_model'] = os.path.join(config_folder_0, 'model_accuracy')		

			# Pretraning B
			config_folder_1 = self.config_folder + "/" + str(K_fold_ind) + "_1"
			if os.path.isdir(config_folder_1):
				shutil.rmtree(config_folder_1)
			os.mkdir(config_folder_1)
			# Submit worker
			submit_job(config_folder_1, parameters, model_params, dimensions, self.K_folds_1, 
				self.train_names_1, self.valid_names_1, self.test_names_1,
				True, False, local, self.logger)

			parameters['pretrained_model'] = os.path.join(config_folder_1, 'model_accuracy')


			# Create fold folder
			config_folder_2 = self.config_folder + "/" + str(K_fold_ind)
			if os.path.isdir(config_folder_2):
				shutil.rmtree(config_folder_2)
			os.mkdir(config_folder_2)
			# Submit worker
			submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold_2,
				self.train_names_2[K_fold_ind], self.valid_names_2[K_fold_ind], self.test_names_2[K_fold_ind],
				save_bool, generate_bool, local, self.logger)
		return