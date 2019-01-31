#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf


def model():
	# return "correction_0"
	# return "Odnade_rnn"
	# return "LSTM_plugged_residual"
	return "seq2seq_0"
	# return "LSTM_affine_OrchCond"

def optimizer():
	return tf.train.AdamOptimizer()
	# return tf.keras.optimizers.Nadam()
	# return tf.train.AdamOptimizer(learning_rate=0.1)
	# return tf.train.GradientDescentOptimizer(0.001)
	# return tf.train.RMSPropOptimizer(0.01)

def import_configs():
	configs= {
		# "residual_basic" : {
		#      'temporal_order' : 5,
		#      'dropout_probability' : 0,
		#      'weight_decay_coeff' : 0,
		#      'MLP_piano_emb': [2000, 2000],
  #       	 'GRU_orch_emb': [2000, 2000],
  #       	 'last_MLP': [],
		#      'tn_weight': 1/10,
		#      'sparsity_coeff': 0,
		# },
		"TEST" : {
		     'temporal_order' : 6,
		     'dropout_probability' : 0,
		     'weight_decay_coeff' : 0,
		     'tn_weight': 1/10,
		     'sparsity_coeff': 0,
		},
		# "NADE" : {
		#      'temporal_order' : 5,
		#      'dropout_probability' : 0,
		#      'weight_decay_coeff' : 0,
		#      'n_hidden_mlp': [],
		#      'n_hidden_gru': [2000, 2000],
		#      'tn_weight': 1/10,
		#      'sparsity_coeff': 0,
		#      'num_ordering': 5,
		# },
		# "correct" : {
		#      'temporal_order' : 5,
		#      'dropout_probability' : 0.3,
		#      'weight_decay_coeff' : 0,
		#      'n_hidden': [2000, 2000],
		#      'tn_weight': 1/10,
		#      'sparsity_coeff': 0,
		#      'mean_iteration_per_note': 5,
		# },
	}
	return configs

def parameters(result_folder):
	return {
		"training_strategy": "trAB_teA",
		"result_folder": result_folder,
		"memory_gpu": 0.95,
		# Data
		"embedded_piano": False,
		"binarize_piano": True,
		"binarize_orchestra": True,
		"duration_piano": False,
		"mask_orch": False,
		"normalizer": "no_normalization",
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
		"debug": {
			"save_measures": False,
		    "summarize": False,
		    "plot_weights": False,
		    "plot_nade_ordering_preds": False,
		    "save_accuracy_along_sampling": False,
		    "salience_embedding": False}
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
	# return "Data_IMSLP__event_level8"
	# return "Data_DEBUG_tempGran8"
	return "Data_bp_bo_noEmb_tempGran32"
	# return "Data_pretraining_tempGran8"

def result_root():
	return "/Users/leo/Recherche/lop/LOP/Results"

def database_root():
	return "/Users/leo/Recherche/databases/Orchestration/LOP_database_06_09_17"

def database_pretraining_root():
	return "/Users/leo/Recherche/databases/Arrangement/SOD"

def local():
	return True