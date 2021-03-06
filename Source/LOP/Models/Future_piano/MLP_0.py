#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.Future_piano.model_lop_future_piano import MLFP

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import Dense

# Hyperopt
from math import log

from LOP.Models.Utils.weight_summary import keras_layer_summary
from LOP.Models.Utils.stacked_rnn import stacked_rnn
from LOP.Utils.hopt_utils import list_log_hopt
from LOP.Utils.hopt_wrapper import qloguniform_int

class MLP_0(MLFP):
    """Recurrent embeddings for both the piano and orchestral scores
    Piano embedding : p(t), ..., p(t+N) through stacked RNN. Last time index of last layer is taken as embedding.
    Orchestra embedding : o(t-N), ..., p(t) Same architecture than piano embedding.
    Then, the concatenation of both embeddings is passed through a MLP
    """
    def __init__(self, model_param, dimensions):

        MLFP.__init__(self, model_param, dimensions)

        # MPL piano
        embeddings_size = model_param['embeddings_size']
        temp = model_param['hs_piano']
        self.hs_piano = list(temp)
        self.hs_piano.append(embeddings_size)

        # MLP orch
        temp = model_param['hs_orch']
        self.hs_orch = list(temp)
        self.hs_orch.append(embeddings_size)

        return

    @staticmethod
    def name():
        return (MLFP.name() + "Recurrent_embeddings_0")
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
    def get_hp_space():
        super_space = MLFP.get_hp_space()

        space = {
            'hs_piano': list_log_hopt(500, 2000, 10, 0, 2, 'hs_piano'),
            'hs_orch': list_log_hopt(500, 2000, 10, 0, 2, 'hs_orch'),
            'embeddings_size': qloguniform_int('hs_orch', log(500), log(1000), 10),
        }

        space.update(super_space)
        return space

    def piano_embedding(self, piano_t, piano_future):
        # Build input
        # Add a time axis to piano_t
        piano_t_time = tf.reshape(piano_t, [-1, 1, self.piano_dim])
        # Concatenate t and future
        input_seq = tf.concat([piano_t_time, piano_future], 1)
        # Flip the matrix along the time axis so that the last time index is t
        input_seq = tf.reshape(piano_t, [-1, self.temporal_order * self.piano_dim])

        with tf.name_scope("MLP"):
            piano_embedding = MLP(x, self.hs_piano, activation='relu')

        return piano_embedding

    def orchestra_embedding(self, orch_past):
        x = tf.reshape(orch_past, [-1, (self.temporal_order-1)*self.orch_dim])
        with tf.name_scope("MLP"):
            piano_embedding = MLP(x, self.hs_orchestra, activation='relu')            
        return orchestra_embedding

    def predict(self, inputs_ph):

        piano_t, _, piano_future, orch_past, _ = inputs_ph
        
        with tf.name_scope("piano_embedding"):
            piano_embedding = self.piano_embedding(piano_t, piano_future)

        with tf.name_scope("orchestra_embedding"):
            orchestra_embedding = self.orchestra_embedding(orch_past)

        #####################
        # Concatenate and predict
        with tf.name_scope("top_layer_prediction_0"):
            top_input = keras.layers.concatenate([orchestra_embedding, piano_embedding], axis=1)
            dense_layer = Dense(1000, activation='relu', name='orch_pred_0')
            top_0 = dense_layer(top_input)
            keras_layer_summary(dense_layer)
        with tf.name_scope("top_layer_prediction_1"):
            dense_layer = Dense(self.orch_dim, activation='sigmoid', name='orch_pred')
            orch_prediction = dense_layer(top_0)
            keras_layer_summary(dense_layer)
        #####################

        return orch_prediction


# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'hs_piano': [500],
# 'hs_orch': [600],
# 'embeddings_size': 500,
