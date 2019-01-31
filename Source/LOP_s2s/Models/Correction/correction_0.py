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

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class correction_0(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		# Architecture
		self.layers = model_param['n_hidden']
		self.recurrent_layers = model_param['n_hidden']
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
		return "correct_trainer"

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
		self.weights["piano_present"] = MLP(self.layers, activation='relu', dropout_probability=self.dropout_probability)
		###############################

		###############################
		# orchestra present
		self.pitch_mask_shape = [self.orch_dim, self.layers[0]]
		first_W = tf.get_variable("first_W", self.pitch_mask_shape, initializer=tf.zeros_initializer())
		first_b = tf.get_variable("first_b", [self.layers[0]], initializer=tf.zeros_initializer())
		self.weights["orch_present"]= [(first_W,first_b)]
		self.weights["orch_present"].append(MLP(self.layers[1:], activation='relu', dropout_probability=self.dropout_probability))
		###############################

		###############################
		# Embedding to prediction

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
			x = orch_t
			for ind_layer, layer in enumerate(self.weights["orch_present"]):
				if ind_layer == 0:
					# Remove pitch from orch and weight
					W, b = layer
					# Mask out the column corresponding to the predicted pitch
					x = tf.matmul(x, tf.multiply(W, pitch_mask)) + b
				else:
					import pdb; pdb.set_trace()
					x = layer(x)
					keras_layer_summary(layer, collections=["weights"])
			orch_t_embedding = x
		return x, x