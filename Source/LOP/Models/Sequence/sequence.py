#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import GRU
from keras.layers import Dense, Dropout, Conv1D

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

from LOP.Models.Utils.weight_summary import keras_layer_summary

class Sequence(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)
        # Hidden layers architecture
        self.n_hs = model_param['n_hidden'] + [int(self.orch_dim)]
        
        return

    @staticmethod
    def name():
        return "Sequence"
    @staticmethod
    def is_keras():
        return True
    @staticmethod
    def optimize():
        return True
    @staticmethod
    def trainer():
        return "sequence_trainer"
    @staticmethod
    def get_hp_space():
        super_space = Model_lop.get_hp_space()
        
        space = {'n_hidden': hp.choice('n_hidden', [
                [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(4000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(4000), 10) for i in range(2)],
                [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(4000), 10) for i in range(3)],
            ]),
        }

        space.update(super_space)
        return space

    def init_weights(self):
        return

    def predict(self, inputs_ph):

        piano_ph, _ = inputs_ph
        #####################
        # GRU for modelling past orchestra
        # First layer
        with tf.name_scope("orch_rnn_0"):
            gru_layer = GRU(self.n_hs[0], return_sequences=True, input_shape=(self.temporal_order, self.piano_dim), activation='relu', dropout=self.dropout_probability)
            x = gru_layer(piano_ph)
            keras_layer_summary(gru_layer, collections=["weights"])

        if len(self.n_hs) > 1:
            # Intermediates layers
            for layer_ind in range(1, len(self.n_hs)):
                with tf.name_scope("orch_rnn_" + str(layer_ind)):
                    gru_layer = GRU(self.n_hs[layer_ind], return_sequences=True, activation='relu', dropout=self.dropout_probability)
                    x = gru_layer(x)
                    keras_layer_summary(gru_layer, collections=["weights"])
        orch_pred = x

        return orch_pred, orch_pred