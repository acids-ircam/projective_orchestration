#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Keras version of the MLP :
# Simple feedforward MLP from the piano input with a binary cross-entropy cost
# Used to test main scripts and as a baseline

from LOP.Models.model_lop import Model_lop
from LOP.Utils.hopt_utils import list_log_hopt
from LOP.Utils.hopt_wrapper import quniform_int
from LOP.Models.Utils.weight_summary import keras_layer_summary

# Tensorflow
import tensorflow as tf
from LOP.Models.Utils.stacked_rnn import stacked_rnn
from LOP.Models.Utils.mlp import MLP
from keras.layers import Dense

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

from LOP.Utils.data_statistics import compute_static_bias_initialization


class correction_NADE_style(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		# Architecture
		self.mlp_piano_present = model_param['mlp_piano_present']
		self.recurrent_layers = model_param['recurrent_layers']
		self.mlp_orch_present = model_param['mlp_orch_present']
		self.mlp_last_pred = model_param['mlp_last_pred']
		# Is it a keras model ?
		self.keras = True
		# Will be computed later
		self.context_embedding_size = None
		# Static bias
		self.static_bias = compute_static_bias_initialization(model_param['activation_ratio'])
		return

	@staticmethod
	def name():
		return "correction_NADE_style"
	@staticmethod
	def binarize_piano():
		return True
	@staticmethod
	def binarize_orchestra():
		return True
	@staticmethod
	def is_keras():
		return True
	@staticmethod
	def optimize():
		return True
	@staticmethod
	def trainer():
		return "correct_trainer_balanced_NADE_style"

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()
		space = {'n_hidden': list_log_hopt(500, 2000, 10, 1, 2, "n_hidden")
		}
		space.update(super_space)
		return space

	def init_weights(self):
		self.weights = {}
		
		###############################
		# Past piano embedding
		self.weights["orch_past"] = stacked_rnn(self.recurrent_layers, activation='relu', dropout_probability=self.dropout_probability)
		###############################

		###############################
		# Past piano embedding
		self.weights["orch_future"] = stacked_rnn(self.recurrent_layers, activation='relu', dropout_probability=self.dropout_probability)
		###############################

		###############################
		# Piano present
		self.weights["piano_present"] = MLP(self.mlp_piano_present, activation='relu', dropout_probability=self.dropout_probability)
		###############################

		###############################
		# orchestra present
		self.weights["orch_present"] = MLP(self.mlp_orch_present, activation='relu', dropout_probability=self.dropout_probability)
		###############################

		###############################
		# Embedding to prediction
		self.weights["final_pred"] = MLP(self.mlp_last_pred, activation='relu', dropout_probability=self.dropout_probability)
		# if len(self.mlp_last_pred) > 0:
		# 	W = tf.get_variable("last_W", [self.orch_dim, self.mlp_last_pred[-1]], initializer=tf.zeros_initializer())
		# else:
		# 	embedding_dim = self.recurrent_layers[-1] * 2 + self.mlp_orch_present[-1] + self.mlp_piano_present[-1]
		# 	W = tf.get_variable("last_W", [embedding_dim, self.orch_dim], initializer=tf.zeros_initializer())
		# b = tf.constant(self.static_bias, dtype=tf.float32, name='precomputed_static_biases')
		# self.weights["final_pred"].append((W,b))
		self.weights["final_pred"].append(Dense(self.orch_dim, activation='sigmoid'))
		###############################
		return

	def predict(self, inputs_ph, pitch_mask):
		piano_t, _, _, orch_t, orch_past, orch_future = inputs_ph

		with tf.name_scope("orch_past"):
			x = orch_past
			for layer in self.weights["orch_past"]:
				x = layer(x)
				keras_layer_summary(layer, collections=["weights"])
			orch_past_embedding = x

		with tf.name_scope("orch_future"):
			x = orch_future
			for layer in self.weights["orch_future"]:
				x = layer(x)
				keras_layer_summary(layer, collections=["weights"])
			orch_future_embedding = x

		with tf.name_scope("present_piano"):
			x = piano_t
			for layer in self.weights["piano_present"]:
				x = layer(x)
				keras_layer_summary(layer, collections=["weights"])
			piano_t_embedding = x

		with tf.name_scope("present_orchestra"):
			with tf.name_scope("build_input"):
				# pitch_mask_reshape = tf.reshape(pitch_mask, [1, self.orch_dim])
				pitch_mask_reshape = pitch_mask
				masked_orch_t = tf.multiply(orch_t, pitch_mask_reshape)
				x = tf.concat([masked_orch_t, pitch_mask_reshape], axis=1)
			for ind_layer, layer in enumerate(self.weights["orch_present"]):	
				x = layer(x)
				keras_layer_summary(layer, collections=["weights"])
			orch_t_embedding = x

		with tf.name_scope("final_pred"):
			# Perhaps concatenate the mask ?
			x = tf.concat([orch_past_embedding, orch_future_embedding, piano_t_embedding, orch_t_embedding], axis=1)
			# for layer_ind in range(len(self.weights["final_pred"])-1):
			# 	layer = self.weights["final_pred"][layer_ind]
			# 	x = layer(x)
			# 	keras_layer_summary(layer, collections=["weights"])
			# W, b = self.weights["final_pred"][-1]
			# pred = tf.sigmoid(tf.matmul(x, W) + b)

			for layer in self.weights["final_pred"]:
				x = layer(x)
				keras_layer_summary(layer, collections=["weights"])
			pred = x

		return pred, pred