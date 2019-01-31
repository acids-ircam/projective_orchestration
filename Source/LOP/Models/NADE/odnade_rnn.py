#!/usr/bin/env python
# -*- coding: utf8 -*-
## Because of shared weight shared accross the graph its more cautious to initialize the weight beforehand
# Keras version of the MLP :
# Simple feedforward MLP from the piano input with a binary cross-entropy cost
# Used to test main scripts and as a baseline

from LOP.Models.model_lop import Model_lop
from LOP.Utils.hopt_utils import list_log_hopt
from LOP.Utils.hopt_wrapper import quniform_int
from LOP.Models.Utils.weight_summary import keras_layer_summary

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers.recurrent import GRU
from keras.layers import Dense, Dropout

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class Odnade_rnn(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		# Hidden layers architecture
		self.layers = model_param['n_hidden_mlp']
		# Hidden layers architecture
		self.n_hs = model_param['n_hidden_gru']
		# Number of different ordering when sampling
		self.num_ordering = model_param['num_ordering']
		# Is it a keras model ?
		self.keras = True
		# Contexxt size will be computed later
		self.context_embedding_size = None
		return

	@staticmethod
	def name():
		return "Odnade_rnn"
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
		return "NADE_informed_trainer"

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()

		space = {'n_hidden': list_log_hopt(500, 2000, 10, 1, 2, "n_hidden"),
			'num_ordering': quniform_int('num_ordering', 5, 5, 1)
		}

		space.update(super_space)
		return space

	def init_weights(self):
		self.weights = {}

		#####################
		# Orchestra embedding: stacked GRU
		if len(self.n_hs) > 1:
			return_sequences = True
		else:
			return_sequences = False
		self.weights["orchestra_emb_GRUs"] = [GRU(self.n_hs[0], return_sequences=return_sequences, input_shape=(self.temporal_order, self.orch_dim),
					activation='relu', dropout=self.dropout_probability)]
		if len(self.n_hs) > 1:
			for layer_ind in range(1, len(self.n_hs)):
				# Last layer ?
				if layer_ind == len(self.n_hs)-1:
					return_sequences = False
				else:
					return_sequences = True
				self.weights["orchestra_emb_GRUs"].append(GRU(self.n_hs[layer_ind], return_sequences=return_sequences,
						activation='relu', dropout=self.dropout_probability))
		#####################

		#####################
		# Piano embedding
		self.weights["piano_emb_MLP"] = Dense(self.n_hs[-1], activation='relu')  # fully-connected layer with 128 units and ReLU activation
		#####################

		#####################
		# NADE part
		self.weights["NADE_mlp"] = []
		for i, l in enumerate(self.layers):
			self.weights["NADE_mlp"].append(Dense(l, activation='relu'))
		self.weights["NADE_mlp_last"] = Dense(self.orch_dim, activation='sigmoid')
		#####################
		return

	def embed_context(self, inputs_ph):
		with tf.name_scope("embed_context"):
			piano_t, _, _, orch_past, _ = inputs_ph

			#####################
			# GRU for modelling past orchestra
			# First layer
			with tf.name_scope("orchestra_past_embedding"):	
				x = orch_past
				for layer_ind, gru_layer in enumerate(self.weights["orchestra_emb_GRUs"]):
					with tf.name_scope("orch_rnn_" + str(layer_ind)):
						x = gru_layer(x)
						keras_layer_summary(gru_layer, collections=["weights"])
				lstm_out = x
			#####################

			#####################
			# gru out and piano(t)
			with tf.name_scope("piano_present_embedding"):
				dense_layer = self.weights["piano_emb_MLP"]
				piano_embedding_ = dense_layer(piano_t)
				piano_embedding = Dropout(self.dropout_probability)(piano_embedding_)
				keras_layer_summary(dense_layer, collections=["weights"])
			#####################

			#####################
			# Merge embeddings        
			with tf.name_scope("merge_embeddings"):
				context_embedding = keras.layers.concatenate([lstm_out, piano_embedding], axis=1)
			#####################

			#####################
			# Context embedding size
			if self.context_embedding_size is None:
				self.context_embedding_size = context_embedding.get_shape()[1]
			#####################

		return context_embedding

	def predict(self, inputs_ph, orch_pred, mask):
		context_embedding = self.embed_context(inputs_ph)
		orch_prediction, x = self.predict_knowing_context(context_embedding, orch_pred, mask)
		return orch_prediction, x

	def predict_knowing_context(self, context_embedding, orch_pred, mask):
		with tf.name_scope("predict_knowing_context"):
			# Build input
			with tf.name_scope("build_input"):
				masked_orch_t = tf.multiply(orch_pred, mask)
				x = tf.concat([context_embedding, masked_orch_t, mask], axis=1)

			# Propagate as in a normal network
			for i, dense in enumerate(self.weights["NADE_mlp"]):
				with tf.name_scope("layer_" + str(i)):
					x = dense(x)
					keras_layer_summary(dense, collections=["weights"])
					x = Dropout(self.dropout_probability)(x)

			# Last layer NADE_mlp
			dense = self.weights["NADE_mlp_last"]
			orch_prediction = dense(x)
			keras_layer_summary(dense, collections=["weights"])

		return orch_prediction, x