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


class Piano_solo(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)
        # Hidden layers architecture
        self.n_hs = model_param['n_hidden']
        return

    @staticmethod
    def name():
        return "Piano_solo"
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
        
        # space = {'n_hidden': hp.choice('n_hidden', [
        #         [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(4000), 10) for i in range(1)],
        #         [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(4000), 10) for i in range(2)],
        #         [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(4000), 10) for i in range(3)],
        #     ]),
        # }
        space = {'n_hidden': [2000,2000]}

        space.update(super_space)
        return space

    def init_weights(self):
        return

    def predict(self, inputs_ph):

        piano_t, _, _, _, _ = inputs_ph
        
        #####################
        # Embedding piano
        with tf.name_scope("piano_embedding"):
            dense_layer = Dense(self.n_hs, activation='relu')  # fully-connected layer with 128 units and ReLU activation
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
            embedding_concat = piano_embedding
            if self.dropout_probability > 0:
                top_input_drop = Dropout(self.dropout_probability)(embedding_concat)
            else:
                top_input_drop = embedding_concat
            # dense_layer = Dense(self.orch_dim, activation='relu', name='orch_pred')
            dense_layer = Dense(self.orch_dim, activation='sigmoid', name='orch_pred')
            orch_prediction = dense_layer(top_input_drop)
            keras_layer_summary(dense_layer, collections=["weights"])
        #####################

        return orch_prediction, embedding_concat

# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'n_hidden': [500, 500],