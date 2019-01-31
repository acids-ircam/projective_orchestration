#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import Dense, Dropout, BatchNormalization, GRU

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

from LOP.Models.Utils.weight_summary import keras_layer_summary
from LOP.Models.Utils.stacked_rnn import stacked_rnn


class FiLM_residual(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)

        return

    @staticmethod
    def name():
        return "FiLM_residual"
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

        space = {}

        space.update(super_space)
        return space

    def init_weights(self):
        self.weights = {}

        self.FilM_dim_0 = 2000
        self.FilM_dim_1 = 2000

        self.weights["FiLM_generator"] = stacked_rnn([2000,2000], 'relu', self.dropout_probability) + [Dense(self.FilM_dim_0*2 + self.FilM_dim_1*2, activation='relu')]

        self.weights["first_layer"] = Dense(2000, activation='relu')

        self.weights["block_0"] = [
            Dense(2000, activation='relu'),
            Dense(self.FilM_dim_0, activation='linear'),
            BatchNormalization()
        ]

        self.weights["block_1"] = [
            Dense(2000, activation='relu'),
            Dense(self.FilM_dim_1, activation='linear'),
            BatchNormalization()
        ]

        self.weights["last_layer"] = Dense(self.orch_dim, activation='softmax')

        return

    def FiLM_generator(self, x, layers):
        for gru_layer in layers:
            x = gru_layer(x)
            keras_layer_summary(gru_layer, collections=["weights"])
        return x

    def residual_FiLM(self, x, gamma, beta, layers):
        original = x
        # Stacked MLPs
        for layer in layers:
            x = layer(x)
            keras_layer_summary(layer, collections=["weights"])
        # FiLM
        x = tf.multiply(gamma, x) + beta
        # Relu
        x = keras.activations.relu(x)
        return x

    def predict(self, inputs_ph):

        piano_t, _, _, orch_past, _ = inputs_ph

        FiLM_Coeff = self.FiLM_generator(orch_past, self.weights['FiLM_generator'])

        # Unpack
        ind_start = 0
        ind_end = ind_start+self.FilM_dim_0
        beta_0 = FiLM_Coeff[:, ind_start:ind_end]
        ind_start = ind_end
        ind_end = ind_start+self.FilM_dim_0
        gamma_0 = FiLM_Coeff[:, ind_start:ind_end]
        ind_start = ind_end
        ind_end = ind_start+self.FilM_dim_1
        beta_1 = FiLM_Coeff[:, ind_start:ind_end]
        ind_start = ind_end
        ind_end = ind_start+self.FilM_dim_1
        gamma_1 = FiLM_Coeff[:, ind_start:ind_end]

        # Adapt input dimension
        first_layer = self.weights["first_layer"]
        x = first_layer(piano_t)
        keras_layer_summary(first_layer, collections=["weights"])

        x = self.residual_FiLM(x, gamma_0, beta_0, self.weights["block_0"])

        x = self.residual_FiLM(x, gamma_1, beta_1, self.weights["block_1"])

        last_layer = self.weights["last_layer"]
        orch_prediction = last_layer(x)
        keras_layer_summary(last_layer, collections=["weights"])

        return orch_prediction, orch_prediction