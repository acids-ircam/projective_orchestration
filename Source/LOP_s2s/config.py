#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf

def model():
	return

def optimizer():
	return tf.train.AdamOptimizer()

def import_configs():
	configs={}
	return configs

def parameters(result_folder):
	return {
		"training_strategy": "trAB_teA",
		"result_folder": result_folder,
		"memory_gpu": 0.95,
		# Data
		"binarize_piano": True,
		"binarize_orchestra": True,
		"duration_piano": False,
		"num_max_contiguous_blocks": int(1e3),
		# Pre-training
		"pretrained_model": None,
		# Train
		"batch_size" : 128,
		"max_iter": 200,                        # nb max of iterations when training 1 configuration of hparams (~200)
		"walltime": 11,                         # in hours, per fold
		# Validation
		"long_range": 5,                        # duration of the long-range prediction
		"k_folds": 10,                          # 0: no k-folds(use only the first fold of a 10-fold (i.e. 8-1-1 split)), -1: leave-one-out
		"min_number_iteration": 3,
		"validation_order": 2,
		"number_strips": 3,
		"overfitting_measure": 'accuracy',
		"save_measures": ['loss', 'accuracy', 'Xent'],
		# Hyperopt
		"max_hyperparam_configs": 20,           # number of hyper-parameter configurations evaluated
		# DEBUG
		"debug": {}
	}

def build_parameters():
    # SHOULD NOT BE MODIFIED BETWEEN RUNS.
    # This defines how much we mix tracks togethers
    # High-value = low mixing rate, more training points
    # Low-value = high mixing rate, less training points (but actually not that much, overall < ~1e3 points)
    return {"chunk_size": 200}

def data_root():
	return "/Users/leo/Recherche/lop/LOP/Data"
def	database_embedding():
	return "/Users/leo/Recherche/databases/Orchestration/Embeddings/embedding_mathieu"
def data_name():
	return
def result_root():
	return "/Users/leo/Recherche/lop/LOP/Results"
def database_root():
	return "/Users/leo/Recherche/databases/Orchestration/LOP_database_06_09_17"
def database_pretraining_root():
	return "/Users/leo/Recherche/databases/Arrangement/SOD"
def local():
	return True