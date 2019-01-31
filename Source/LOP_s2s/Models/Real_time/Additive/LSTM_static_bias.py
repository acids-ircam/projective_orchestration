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

    def predict(self, inputs_ph):
        
        piano_t, _, _, orch_past, _ = inputs_ph
        
        #####################
        # Batch norm
        # if self.binarize_piano:
        #   piano_t = BatchNorm()
        #####################


        #####################
        # GRU for modelling past orchestra
        # First layer
        if len(self.n_hs) > 1:
            return_sequences = True
        else:
            return_sequences = False
        
        with tf.name_scope("orch_rnn_0"):
            gru_layer = GRU(self.n_hs[0], return_sequences=return_sequences, input_shape=(self.temporal_order, self.orch_dim),
                    activation='relu', dropout=self.dropout_probability)
            x = gru_layer(orch_past)
            keras_layer_summary(gru_layer)
        
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
                    keras_layer_summary(gru_layer)

        lstm_out = x
        #####################
        
        #####################
        # gru out and piano(t)
        with tf.name_scope("piano_embedding"):
            # fully-connected layer with 128 units and ReLU activation
            piano_t_drop = Dropout(self.dropout_probability)(piano_t)
            dense_layer = Dense(self.n_hs[-1], activation='relu')
            piano_embedding = dense_layer(piano_t_drop)
            keras_layer_summary(dense_layer)
        #####################

        #####################
        # Concatenate and predict
        with tf.name_scope("top_layer_prediction"):
            top_input = keras.layers.concatenate([lstm_out, piano_embedding], axis=1)
            top_input_drop = Dropout(self.dropout_probability)(top_input)
            # First, just the linear part
            dense_layer = Dense(self.orch_dim, activation='linear', name='orch_pred')
            biased_linear_pred = dense_layer(top_input_drop)
            keras_layer_summary(dense_layer)
            # Add the pre-computed static biases
            precomputed_biases = tf.constant(self.static_bias, dtype=tf.float32, name='precomputed_static_biases')
            fix_linear_pred = biased_linear_pred +  precomputed_biases
            # Now pass in the non-linearity
            orch_prediction = Activation('sigmoid')(fix_linear_pred)
        #####################
        
        embedding_concat = top_input
        return orch_prediction, embedding_concat


# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'n_hidden': [500, 500],
