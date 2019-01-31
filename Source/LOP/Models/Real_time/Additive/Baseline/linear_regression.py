#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Simple Linear regression in Tensorflow
# Main purposes:
# 	- baseline for a very simple model
# 	- speed test for loading data: if the training time of this model is almost the same as for a more complex model (even simple like MLP), then it means that the input data pipeline is the bottleneck

from LOP.Models.model_lop import Model_lop
from LOP.Utils.hopt_utils import list_log_hopt
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


class Linear_regression(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		return

	@staticmethod
	def name():
		return "Linear_regression"
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
	def trainer():
		return "standard_trainer"

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()
		return super_space

	def init_weights(self):
		return

	def predict(self, inputs_ph):
		piano_t, _, _, orch_past, _ = inputs_ph
		W = tf.get_variable("W", (self.piano_dim, self.orch_dim), initializer=tf.random_normal_initializer())
		orch_prediction = tf.matmul(piano_t, W)
		return orch_prediction, None