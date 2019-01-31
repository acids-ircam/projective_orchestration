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


class correction_0(Model_lop):
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
		return

	@staticmethod
	def name():
		return "correction_0"
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
		return "correct_trainer_balanced"

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
		W_shape = [self.orch_dim, self.mlp_orch_present[0]]
		first_W = tf.get_variable("first_W", W_shape, initializer=tf.zeros_initializer())
		first_b = tf.get_variable("first_b", [self.mlp_orch_present[0]], initializer=tf.zeros_initializer())
		self.weights["orch_present"]= [(first_W, first_b)]
		if len(self.mlp_orch_present) > 1:
			self.weights["orch_present"].extend(MLP(self.mlp_orch_present[1:], activation='relu', dropout_probability=self.dropout_probability))
		###############################

		###############################
		# Embedding to prediction
		self.weights["final_pred"] = MLP(self.mlp_last_pred, activation='relu', dropout_probability=self.dropout_probability)
		self.weights["final_pred"].append(Dense(1, activation='sigmoid'))
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
			for ind_layer, layer in enumerate(self.weights["orch_present"]):
				if ind_layer == 0:
					# Remove pitch from orch and weight
					W, b = layer
					with tf.name_scope("masking_orch_t"):
						orch_t_masked = tf.boolean_mask(orch_t, pitch_mask, axis=1)
						W_masked = tf.boolean_mask(W, pitch_mask, axis=0)
					# Mask out the column corresponding to the predicted pitch
					x = tf.matmul(orch_t_masked, W_masked) + b
				else:
					x = layer(x)
					keras_layer_summary(layer, collections=["weights"])
			orch_t_embedding = x

		with tf.name_scope("final_pred"):
			# Perhaps concatenate the mask ?
			x = tf.concat([orch_past_embedding, orch_future_embedding, piano_t_embedding, orch_t_embedding], axis=1)
			for layer in self.weights["final_pred"]:
				x = layer(x)
				keras_layer_summary(layer, collections=["weights"])
			pred = x

		# pred has size (batch, 1)
		return pred, pred