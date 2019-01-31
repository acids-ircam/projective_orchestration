#!/usr/bin/env python
# -*- coding: utf8 -*-

""" This is the script called by qsub on Guillimin
"""

import os
import sys
import time
import shutil
import csv
import numpy as np
import pickle as pkl
from LOP.Scripts.import_functions import import_model, import_normalizer
from LOP.Scripts.train import train
import LOP.Scripts.generations.forward as forward
import LOP.Utils.data_statistics as data_statistics

def train_wrapper(parameters, model_params,
	dimensions, config_folder_fold, K_fold,
	track_paths_generation, 
	save_model, generate_bool, logger):
	
	Model = import_model.import_model(parameters["model_name"])

	train_folds = K_fold['train']
	valid_folds = K_fold['valid']
	test_folds = K_fold['test']
	
	# Instanciate a normalizer for the input
	# normalizer = Normalizer(train_folds, n_components=20, whiten=True, parameters=parameters)
	# normalizer = Normalizer(train_folds, parameters)
	normalizer = import_normalizer.import_normalizer(parameters["normalizer"], dimensions, train_folds, parameters)
	pkl.dump(normalizer, open(os.path.join(config_folder_fold, 'normalizer.pkl'), 'wb'))
	if not parameters["embedded_piano"]:
		# Normalization is by-passed when embedding is used
		dimensions['piano_input_dim'] = normalizer.norm_dim

	# Compute training data's statistics for improving learning (e.g. weighted Xent)
	time_data_stats_0 = time.time()
	activation_ratio = data_statistics.get_activation_ratio(train_folds, dimensions['orch_dim'], parameters)
	mean_number_units_on = data_statistics.get_mean_number_units_on(train_folds, parameters)
	_, mask_inter_orch_NADE = data_statistics.get_mask_inter_orch_NADE(train_folds, dimensions['orch_dim'], parameters)
	_, mask_piano_orch_NADE = data_statistics.get_mask_piano_orch_NADE(train_folds, dimensions['piano_embedded_dim'], dimensions['orch_dim'], parameters)	
	
	# On all data
	# all_folds = train_folds + valid_folds + test_folds
	# activation_ratio_ALL = data_statistics.get_activation_ratio(all_folds, dimensions['orch_dim'], parameters)
	# mean_number_units_on_ALL = data_statistics.get_mean_number_units_on(all_folds, parameters)
	# num_coOcc_orch_ALL, mask_inter_orch_NADE_ALL = data_statistics.get_mask_inter_orch_NADE(all_folds, dimensions['orch_dim'], parameters)
	# num_coOcc_piano_ALL, mask_piano_orch_NADE_ALL = data_statistics.get_mask_piano_orch_NADE(all_folds, dimensions['piano_embedded_dim'], dimensions['orch_dim'], parameters)
	# orch_semi_tone = 0
	# for i in range(dimensions['orch_dim']-1):
	# 	orch_semi_tone += num_coOcc_orch_ALL[i,i+1]

	# It's okay to add this value to the parameters now because we don't need it for persistency, 
	# this is only training regularization
	model_params['activation_ratio'] = activation_ratio
	parameters['activation_ratio'] = activation_ratio
	parameters['mask_inter_orch_NADE'] = mask_inter_orch_NADE
	parameters['mask_piano_orch_NADE'] = mask_piano_orch_NADE
	model_params['mean_number_units_on'] = mean_number_units_on
	time_data_stats_1 = time.time()
	logger.info('TTT : Computing data statistics took {} seconds'.format(time_data_stats_1-time_data_stats_0))
	
	########################################################
	# Persistency
	pkl.dump(model_params, open(config_folder_fold + '/model_params.pkl', 'wb'))
	pkl.dump(Model.is_keras(), open(config_folder_fold + '/is_keras.pkl', 'wb'))
	pkl.dump(parameters, open(config_folder_fold + '/script_parameters.pkl', 'wb'))

	############################################################
	# Update train_param and model_param dicts with new information from load data
	############################################################
	def count_batch(fold):
		counter_batch = 0
		counter_points = 0
		for chunk in fold:
			counter_batch += len(chunk["batches"])
			for batch in chunk["batches"]:
				counter_points += len(batch)
		return counter_batch, counter_points
	n_train_batches, _ = count_batch(train_folds)
	n_val_batches, n_val_points = count_batch(valid_folds)
	n_test_batches, n_test_points = count_batch(test_folds)

	logger.info('# Num train batch :  {}'.format(n_train_batches))
	logger.info('# Num val batch :  {}'.format(n_val_batches))
	logger.info('# Num test batch :  {}'.format(n_test_batches))

	parameters['n_train_batches'] = n_train_batches
	parameters['n_val_batches'] = n_val_batches
	parameters['n_test_batches'] = n_test_batches

	############################################################
	# Instanciate model and save folder
	############################################################
	model = Model(model_params, dimensions)
	for measure_name in parameters["save_measures"]:
		os.mkdir(config_folder_fold + '/model_' + measure_name)
	
	############################################################
	# Train
	############################################################
	time_train_0 = time.time()
	accuracy_training, valid_tabs, test_score, best_epoch, valid_tabs_sampled, test_score_sampled, best_epoch_sampled, valid_tabs_LR, test_score_LR, best_epoch_LR =\
		train(model, train_folds, valid_folds, test_folds, normalizer, model_params, parameters, config_folder_fold, time_train_0, logger)
	time_train_1 = time.time()
	training_time = time_train_1-time_train_0
	logger.info('TTT : Training data took {} seconds'.format(training_time))
	logger.info('# Best loss obtained at epoch :  {}'.format(best_epoch['loss']))
	logger.info('# Loss :  {}'.format(valid_tabs['loss'][best_epoch['loss']]))
	logger.info('# Accuracy :  {}'.format(valid_tabs['accuracy'][best_epoch['accuracy']]))
	logger.info('###################\n')

	############################################################
	# Write result in a txt file
	############################################################
	os.mkdir(os.path.join(config_folder_fold, 'results_short_range'))
	os.mkdir(os.path.join(config_folder_fold, 'results_sampled'))
	os.mkdir(os.path.join(config_folder_fold, 'results_long_range'))
	
	# Accuracy_training
	np.savetxt(os.path.join(config_folder_fold, 'results_short_range/accuracy_training.txt'), accuracy_training, fmt='%.6f')

	# Short range
	for measure_name, measure_curve in valid_tabs.items():
		np.savetxt(os.path.join(config_folder_fold, 'results_short_range', measure_name + '.txt'), measure_curve, fmt='%.6f')
		with open(os.path.join(config_folder_fold, 'results_short_range', measure_name + '_best_epoch.txt'), 'w') as f:
			f.write("{:d}".format(best_epoch[measure_name]))

	# Sampled
	for measure_name, measure_curve in valid_tabs_sampled.items():
		np.savetxt(os.path.join(config_folder_fold, 'results_sampled', measure_name + '.txt'), measure_curve, fmt='%.6f')
		with open(os.path.join(config_folder_fold, 'results_sampled', measure_name + '_best_epoch.txt'), 'w') as f:
			f.write("{:d}".format(best_epoch[measure_name]))

	# Long range
	for measure_name, measure_curve in valid_tabs_LR.items():
		np.savetxt(os.path.join(config_folder_fold, 'results_long_range', measure_name + '.txt'), measure_curve, fmt='%.6f')
		with open(os.path.join(config_folder_fold, 'results_long_range', measure_name + '_best_epoch.txt'), 'w') as f:
			f.write("{:d}".format(best_epoch[measure_name]))

	# Test scores
	with open(os.path.join(config_folder_fold, 'test_score.csv'), 'w') as csvfile:
		fieldnames = test_score.keys()
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
		writer.writeheader()
		writer.writerow(test_score)
	with open(os.path.join(config_folder_fold, 'test_score_sampled.csv'), 'w') as csvfile:
		fieldnames = test_score_sampled.keys()
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
		writer.writeheader()
		writer.writerow(test_score_sampled)
	with open(os.path.join(config_folder_fold, 'test_score_LR.csv'), 'w') as csvfile:
		fieldnames = test_score_LR.keys()
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
		writer.writeheader()
		writer.writerow(test_score_LR)

	# Generating
	if generate_bool:
		generate_wrapper(config_folder_fold, track_paths_generation, logger)
	if not save_model:
		for measure_name in parameters['save_measures']:
			shutil.rmtree(config_folder_fold + '/model_' + measure_name)
		########################################################

	# Write DONE in the config folder
	open(config_folder_fold + "/DONE", 'w').close()
	return

def generate_wrapper(config_folder_fold, track_paths_generation, logger):
	for score_source in track_paths_generation:
			save_folder = config_folder_fold + '/generations'
			forward.generate_midi(config_folder_fold, score_source, save_folder, initialization_type="seed", number_of_version=3, duration_gen=100, logger_generate=logger)
	return


if __name__ == '__main__':

	config_folder_fold = sys.argv[1]
	context_folder = config_folder_fold + "/context"
	
	# Get parameters
	parameters =  pkl.load(open(context_folder + "/parameters.pkl","rb"))
	model_params = pkl.load(open(context_folder + "/model_params.pkl","rb"))
	dimensions = pkl.load(open(context_folder + "/dimensions.pkl","rb")) 
	K_fold = pkl.load(open(context_folder + "/K_fold.pkl","rb"))
	track_paths_generation = pkl.load(open(context_folder + "/track_paths_generation.pkl","rb"))
	save_bool = pkl.load(open(context_folder + "/save_bool.pkl","rb"))
	generate_bool = pkl.load(open(context_folder + "/generate_bool.pkl","rb"))

	import logging
	logger = logging.getLogger('worker')
	logger.setLevel(logging.INFO)
	hdlr = logging.FileHandler(config_folder_fold + '/log.txt')	
	hdlr.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)

	train_wrapper(parameters, model_params,
		dimensions, config_folder_fold, K_fold,
		track_paths_generation, 
		save_bool, generate_bool, logger)