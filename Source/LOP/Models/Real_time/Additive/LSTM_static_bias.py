#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:29:28 2017

@author: leo
"""

#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers.recurrent import GRU
from keras.layers import Dense, Activation, Dropout

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp
from LOP.Utils.data_statistics import compute_static_bias_initialization

from LOP.Models.Utils.weight_summary import keras_layer_summary

class LSTM_static_bias(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		# Hidden layers architecture
		self.n_hs = model_param['n_hidden']
		self.static_bias = compute_static_bias_initialization(model_param['activation_ratio'])
		return

	@staticmethod
	def name():
		return "LSTM_static_bias"
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
		return "standard_trainer"
	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()

		space = {'n_hidden': hp.choice('n_hidden', [
				[hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(5000), 10) for i in range(1)],
				[hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(5000), 10) for i in range(2)],
				[hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(5000), 10) for i in range(3)],
			]),
		}

		space.update(super_space)
		return space

	def init_weights(self):
		return

	def predict(self, inputs_ph):
		
		piano_t, _, _, orch_past, _ = inputs_ph

		#####################
		# GRU for modelling past orchestra
		# First layer
		with tf.name_scope("orch_embedding"):
			if len(self.n_hs) > 1:
				return_sequences = True
			else:
				return_sequences = False
			
			with tf.name_scope("orch_rnn_0"):
				gru_layer = GRU(self.n_hs[0], return_sequences=return_sequences, input_shape=(self.temporal_order, self.orch_dim),
						activation='relu', dropout=self.dropout_probability)
				x = gru_layer(orch_past)
				keras_layer_summary(gru_layer, collections=["weights"])
			
			if len(self.n_hs) > 1:
				# Intermediates layers
				for layer_ind in range(1, len(self.n_hs)):
					# Last layer ?
					if layer_ind == len(self.n_hs)-1:
						return_sequences = False
					else:
						return_sequences = True
					with tf.name_scope("orch_rnn_" + str(layer_ind)):
						gru_layer = GRU(self.n_hs[layer_ind], return_sequences=return_sequences,
								activation='relu', dropout=self.dropout_probability)
						x = gru_layer(x)
						keras_layer_summary(gru_layer, collections=["weights"])

			lstm_out = x
		#####################
		
		#####################
		# Embedding piano
		with tf.name_scope("piano_embedding"):
			dense_layer = Dense(self.n_hs[-1], activation='relu')  # fully-connected layer with 128 units and ReLU activation
			piano_embedding_ = dense_layer(piano_t)
			if self.dropout_probability > 0:
				piano_embedding = Dropout(self.dropout_probability)(piano_embedding_)
			else:
				piano_embedding = piano_embedding_
			keras_layer_summary(dense_layer, collections=["weights"])

			# num_filter = 30
			# conv_layer = Conv1D(num_filter, 12, activation='relu')
			# piano_t_reshape = tf.reshape(piano_t, [-1, self.piano_dim, 1])
			# piano_embedding = conv_layer(piano_t_reshape)
			# piano_embedding = tf.reshape(piano_embedding, [-1, (self.piano_dim-12+1) * num_filter])
			# keras_layer_summary(conv_layer, collections=["weights"])
		#####################

		#####################
		# Concatenate and predict
		with tf.name_scope("top_layer_prediction"):
			embedding_concat = keras.layers.concatenate([lstm_out, piano_embedding], axis=1)
			if self.dropout_probability > 0:
				top_input_drop = Dropout(self.dropout_probability)(embedding_concat)
			else:
				top_input_drop = embedding_concat
			# dense_layer = Dense(self.orch_dim, activation='relu', name='orch_pred')
			dense_layer = Dense(self.orch_dim, activation='linear', name='orch_pred', use_bias=True)
			orch_prediction = dense_layer(top_input_drop)
			keras_layer_summary(dense_layer, collections=["weights"])
			orch_prediction = Activation("sigmoid")(orch_prediction + self.static_bias)
		#####################

		return orch_prediction, embedding_concat


# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'n_hidden': [500, 500],
