#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Keras version of the MLP :
# Simple feedforward MLP from the piano input with a binary cross-entropy cost
# Used to test main scripts and as a baseline
# 
# Instead of concatenating the mask, now we simply apply -0.5 to the input vector, and multiply by the binary mask
# Hence, note off = -0.5, note on = 0.5, note masked = 0

from LOP.Models.model_lop import Model_lop
from LOP.Utils.hopt_utils import list_log_hopt
from LOP.Utils.hopt_wrapper import quniform_int
from LOP.Models.Utils.weight_summary import keras_layer_summary

# Tensorflow
import tensorflow as tf

# Keras
from keras import regularizers
from keras.layers import Dense, Dropout

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class Odnade_mlp_3(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		# Hidden layers architecture
		self.layers_embedding = model_param['n_hidden_embedding']
		# Hidden layers architecture
		self.layers_NADE = model_param['n_hidden_NADE']
		# Number of different ordering when sampling
		self.num_ordering = model_param['num_ordering']
		# Is it a keras model ?
		#self.keras = True
		# Will be computed later
		self.context_embedding_size = None
		return

	@staticmethod
	def name():
		return "Odnade_mlp_3"
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
		# return "NADE_Gibbs_trainer"
		# return "NADE_informed_trainer"
		return "NADE_trainer"

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()
		space = {
			'n_hidden_embedding': hp.choice('n_hidden_embedding', [
                [hopt_wrapper.qloguniform_int('n_hidden_embedding_'+str(i), log(1500), log(3000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_embedding_'+str(i), log(1500), log(3000), 10) for i in range(2)],
                [hopt_wrapper.qloguniform_int('n_hidden_embedding_'+str(i), log(1500), log(3000), 10) for i in range(3)],
            ]),
            'n_hidden_NADE': hp.choice('n_hidden_NADE', [
                [hopt_wrapper.qloguniform_int('n_hidden_NADE_'+str(i), log(1500), log(3000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_NADE_'+str(i), log(1500), log(3000), 10) for i in range(2)],
                [hopt_wrapper.qloguniform_int('n_hidden_NADE_'+str(i), log(1500), log(3000), 10) for i in range(3)],
            ]),
			'num_ordering': quniform_int('num_ordering', 5, 10, 1)
		}
		space.update(super_space)
		return space

	def init_weights(self):
		self.weights = {}
		
		self.weights["MLP_embed"] = []
		for i, l in enumerate(self.layers_embedding):
			with tf.name_scope("layer_" + str(i)):
				self.weights["MLP_embed"].append(Dense(l, activation='relu'))

		self.weights["MLP_NADE"] = []
		for i, l in enumerate(self.layers_NADE):
			with tf.name_scope("layer_" + str(i)):
				self.weights["MLP_NADE"].append(Dense(l, activation='relu'))

		self.weights["last_MLP"] = Dense(self.orch_dim, activation='sigmoid')
		return

	def embed_context(self, inputs_ph):
		# MLP on concatenation of past orchestra and present piano
		with tf.name_scope("embed_context"):
			piano_t, _, _, orch_past, _ = inputs_ph

			orch_past_flat = tf.reshape(orch_past, [-1, (self.temporal_order-1) * self.orch_dim])
			x = tf.concat([piano_t, orch_past_flat], axis=1)

			for dense in self.weights["MLP_embed"]:
				x = dense(x)
				keras_layer_summary(dense, collections=["weights"])
				if self.dropout_probability > 0:
					x = Dropout(self.dropout_probability)(x)

			context_embedding = x
			if self.context_embedding_size is None:
				self.context_embedding_size = context_embedding.get_shape()[1]

		return context_embedding

	def predict(self, inputs_ph, orch_pred, mask):
		context_embedding = self.embed_context(inputs_ph)
		orch_prediction, x = self.predict_knowing_context(context_embedding, orch_pred, mask)
		return orch_prediction, x
	
	def predict_knowing_context(self, context_embedding_precomputed, orch_pred, mask):
		with tf.name_scope("predict_knowing_context"):
			# Build input
			x = tf.multiply(orch_pred - 0.5, mask)
			for dense in self.weights["MLP_NADE"]:
				x = dense(x)
				keras_layer_summary(dense, collections=["weights"])
				if self.dropout_probability > 0:
					x = Dropout(self.dropout_probability)(x)

			# concatenate
			x = tf.concat([context_embedding_precomputed, x], axis=1)
			dense = self.weights["last_MLP"]
			orch_prediction = dense(x)
			keras_layer_summary(dense, collections=["weights"])

		return orch_prediction, x