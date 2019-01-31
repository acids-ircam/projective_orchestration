#!/usr/bin/env python
# -*- coding: utf8 -*-

# Wrapper for stacked recurrent nets

import tensorflow as tf
import keras
from keras.layers.recurrent import GRU, LSTM
from LOP.Models.Utils.weight_summary import keras_layer_summary


def stacked_rnn(layers, activation, dropout_probability):

	# def layer_rnn(layer, rnn_type, return_sequences):
	# 	if rnn_type is 'gru':
	# 		this_layer = GRU(layer, return_sequences=return_sequences,
	# 				activation=activation, dropout=dropout_probability)
	# 	return this_layer

	if len(layers) > 1:
		return_sequences = True
	else:
		return_sequences = False
	
	ret = [GRU(layers[0], return_sequences=return_sequences, activation=activation, dropout=dropout_probability)]
	
	if len(layers) > 1:
		for layer_ind in range(1, len(layers)):
			# Last layer ?
			if layer_ind == len(layers)-1:
				return_sequences = False
			else:
				return_sequences = True
			ret.append(GRU(layers[layer_ind], return_sequences=return_sequences, activation=activation, dropout=dropout_probability))
	return ret
