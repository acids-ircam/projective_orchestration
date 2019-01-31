#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from keras.layers import Dense, Dropout
from LOP.Models.Utils.weight_summary import keras_layer_summary

def MLP(layers, activation='relu', dropout_probability=0):
	ret = []
	for layer_ind, num_unit in enumerate(layers):
		with tf.variable_scope(str(layer_ind)):
			ret.append(Dense(num_unit, activation=activation))
			if dropout_probability > 0:
				ret.append(Dropout(dropout_probability))
	return ret