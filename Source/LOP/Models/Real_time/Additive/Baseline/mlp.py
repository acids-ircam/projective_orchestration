#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Simple feedforward MLP from the piano input with a binary cross-entropy cost
# Used to test main scripts and as a baseline

from LOP.Models.model_lop import Model_lop
from LOP.Models.Utils.weight_summary import variable_summary

# Tensorflow
import tensorflow as tf

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class MLP(Model_lop):
	def __init__(self, model_param, dimensions):

		Model_lop.__init__(self, model_param, dimensions)

		# Hidden layers architecture
		self.layers = [self.piano_dim] + list(model_param['n_hidden'])
		# Is it a keras model ?
		self.keras = False

		return

	@staticmethod
	def name():
		return "Baseline_MLP"
	@staticmethod
	def binarize_piano():
		return True
	@staticmethod
	def binarize_orchestra():
		return True
	@staticmethod
	def is_keras():
		return False
	@staticmethod
	def optimize():
		return True

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()

		space = {'n_hidden': hp.choice('n_hidden', [
				[],
				[hopt_wrapper.qloguniform_int('layer_'+str(i), log(100), log(5000), 10) for i in range(1)],
				[hopt_wrapper.qloguniform_int('layer_'+str(i), log(100), log(5000), 10) for i in range(2)],
				[hopt_wrapper.qloguniform_int('layer_'+str(i), log(100), log(5000), 10) for i in range(3)],
			]),
		}

		space.update(super_space)
		return space

	def predict(self, inputs_ph):
		piano_t, _, _, orch_past, _ = inputs_ph
		def add_layer(input, W_shape, b_shape, sigmoid=False):
			W = tf.get_variable("W", W_shape, initializer=tf.random_normal_initializer())
			b = tf.get_variable("b", b_shape, initializer=tf.constant_initializer(0.0))
			if sigmoid:
				out = tf.nn.sigmoid(tf.matmul(x, W) + b)
			else:
				out = tf.nn.relu(tf.matmul(x, W) + b)
			return out

		x = piano_t
		
		for l in range(len(self.layers)-1):
			with tf.variable_scope("layer_" + str(l)):
				x = add_layer(x, [self.layers[l], self.layers[l+1]], [self.layers[l+1]])

		with tf.variable_scope("layer_" + str(len(self.layers))):
			orch_prediction = add_layer(x, [self.layers[-1], self.orch_dim], [self.orch_dim], True)

		return orch_prediction, None

# "0" : {
#  'temporal_order' : 3,
#  'dropout_probability' : 0,
#  'weight_decay_coeff' : 0,
#  'n_hidden': [200, 200],
#  'tn_weight': 1/10,
#  'sparsity_coeff': 0
# },