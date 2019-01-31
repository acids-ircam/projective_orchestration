#!/usr/bin/env python
# -*- coding: utf8 -*-


from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class Model_lop(object):
	
	def __init__(self, model_param, dimensions):
		# Dimensions
		self.temporal_order = dimensions['temporal_order']
		self.piano_dim = dimensions['piano_input_dim'] 	# Piano passed to Data are embedded then normalized, so we want piano_norm_dim as the input data
		self.orch_dim = dimensions['orch_dim']

		# Regularization paramters
		self.dropout_probability = model_param['dropout_probability']
		self.weight_decay_coeff = model_param['weight_decay_coeff']

		########################
		# EXPERIMENTATIONS 
		self.tn_weight = model_param['tn_weight']
		self.sparsity_coeff = model_param['sparsity_coeff']
		########################

		self.params = []
		return

	@staticmethod
	def get_hp_space():
		space_training = {
			# 'temporal_order': hopt_wrapper.qloguniform_int('temporal_order', log(2), log(7), 1),
			'temporal_order': hopt_wrapper.qloguniform_int('temporal_order', log(5), log(5), 1),
			'tn_weight': 1/10,
			'sparsity_coeff': 0,
		}

		space_regularization = {
			'dropout_probability': hp.choice('dropout', 
				[0.0]),
			'weight_decay_coeff': hp.choice('weight_decay_coeff', 
				[0.0])
		}

		space_training.update(space_regularization)
		return space_training