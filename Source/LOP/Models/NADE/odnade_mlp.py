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

# Keras
from keras import regularizers
from keras.layers import Dense, Dropout

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class Odnade_mlp(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		# Hidden layers architecture
		self.layers = model_param['n_hidden']
		# Number of different ordering when sampling
		self.num_ordering = model_param['num_ordering']
		# Is it a keras model ?
		#self.keras = True
		# Will be computed later
		self.context_embedding_size = None
		return

	@staticmethod
	def name():
		return "Odnade_mlp"
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
		return "NADE_trainer"

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()
		space = {
			'n_hidden': hp.choice('n_hidden', [
                [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(1500), log(3000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(1500), log(3000), 10) for i in range(2)],
                [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(1500), log(3000), 10) for i in range(3)],
            ]),
			'num_ordering': quniform_int('num_ordering', 5, 10, 1)
		}
		space.update(super_space)
		return space

	def init_weights(self):
		self.weights = {}
		self.weights["MLP"] = []
		for i, l in enumerate(self.layers):
			with tf.name_scope("layer_" + str(i)):
				self.weights["MLP"].append(Dense(l, activation='relu'))

		self.weights["last_MLP"] = Dense(self.orch_dim, activation='sigmoid')
		return

	def embed_context(self, inputs_ph):
		with tf.name_scope("embed_context"):
			piano_t, _, _, orch_past, _ = inputs_ph

			orch_past_flat = tf.reshape(orch_past, [-1, (self.temporal_order-1) * self.orch_dim])
			context_embedding = tf.concat([piano_t, orch_past_flat], axis=1)

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
			with tf.name_scope("build_input"):
				masked_orch_t = tf.multiply(orch_pred, mask)
				x = tf.concat([context_embedding_precomputed, masked_orch_t, mask], axis=1)

			with tf.name_scope("NADE_prediction"):
				# Propagate as in a normal network
				for dense in self.weights["MLP"]:
					x = dense(x)
					keras_layer_summary(dense, collections=["weights"])
					x = Dropout(self.dropout_probability)(x)

				dense = self.weights["last_MLP"]
				orch_prediction = dense(x)
				keras_layer_summary(dense, collections=["weights"])

		return orch_prediction, x