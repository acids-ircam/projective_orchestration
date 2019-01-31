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


class LSTM_affine_OrchCond(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)
        self.rnns = model_param['n_hidden']
        self.film_dim = model_param['film_dim']

        return

    @staticmethod
    def name():
        return "LSTM_affine_OrchCond"
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

        space = {
            'n_hidden': hp.choice('n_hidden', [
                [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(1500), log(3000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(1500), log(3000), 10) for i in range(2)],
                [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(1500), log(3000), 10) for i in range(3)],
            ]),
            'film_dim': hopt_wrapper.qloguniform_int('film_dim', log(1000), log(2000), 10),
        }

        space.update(super_space)
        return space

    def init_weights(self):
        self.weights = {}

        self.weights["MLP"] = Dense(self.film_dim, activation='relu')

        self.weights["FiLM_generator"] = stacked_rnn(self.rnns, 'relu', self.dropout_probability)
        self.weights["gamma"] = Dense(self.film_dim, activation='sigmoid')
        self.weights["beta"] = Dense(self.film_dim, activation='relu')

        self.weights["last_layer"] = Dense(self.orch_dim, activation='softmax')

        return

    def FiLM_generator(self, x, layers):
        for gru_layer in layers:
            x = gru_layer(x)
            keras_layer_summary(gru_layer, collections=["weights"])

        gamma_layer = self.weights["gamma"]
        beta_layer = self.weights["beta"]
        keras_layer_summary(gamma_layer, collections=["weights"])
        keras_layer_summary(beta_layer, collections=["weights"])

        gamma = gamma_layer(x)
        beta = beta_layer(x)

        return gamma, beta

    def predict(self, inputs_ph):

        piano_t, _, _, orch_past, _ = inputs_ph

        gamma, beta = self.FiLM_generator(orch_past, self.weights['FiLM_generator'])

        # Unpack
        ind_end = 0
        x = piano_t
        layer = self.weights["MLP"]
        x = layer(x)
        keras_layer_summary(layer, collections=["weights"])
        
        # FiLM layer
        x = tf.multiply(gamma, x) + beta
        x = keras.activations.relu(x)

        last_layer = self.weights["last_layer"]
        orch_prediction = last_layer(x)
        keras_layer_summary(last_layer, collections=["weights"])

        return orch_prediction, orch_prediction