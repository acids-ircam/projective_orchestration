#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.Future_past_piano.model_lop_future_past_piano import MLFPP

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import Dense, GRU, Dropout
from LOP.Models.Utils.weight_summary import keras_layer_summary

# Hyperopt
from LOP.Utils import hopt_wrapper
from hyperopt import hp
from math import log

# Architecture
from LOP.Models.Utils.stacked_rnn import stacked_rnn
from LOP.Models.Utils.mlp import MLP
from LOP.Utils.hopt_wrapper import qloguniform_int

class Future_past_piano(MLFPP):
    """Recurrent embeddings for both the piano and orchestral scores
    Piano embedding : p(t), ..., p(t+N) through stacked RNN. Last time index of last layer is taken as embedding.
    Orchestra embedding : o(t-N), ..., p(t) Same architecture than piano embedding.
    Then, the concatenation of both embeddings is passed through a MLP
    """
    def __init__(self, model_param, dimensions):
        MLFPP.__init__(self, model_param, dimensions)
        # Gru piano
        self.mlp_piano = model_param['mlp_piano']
        self.recurrent_piano = model_param["recurrent_piano"]
        self.recurrent_orch = model_param['recurrent_orch']
        # self.last_mlp = model_param['last_mlp']
        return

    @staticmethod
    def name():
        return ("future_past_piano")
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
        super_space = MLFPP.get_hp_space()

        space = {'recurrent_orch': hp.choice('n_hidden', [
                [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(500), log(3000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(500), log(3000), 10) for i in range(2)],
                [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(500), log(3000), 10) for i in range(3)],
                ]),
            'recurrent_piano': hp.choice('n_hidden', [
                [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(500), log(3000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(500), log(3000), 10) for i in range(2)],
                ]),
            'mlp_piano': hp.choice('n_hidden', [
                [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(500), log(3000), 10) for i in range(1)],
                [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(500), log(3000), 10) for i in range(2)],
                ])
        }

        space.update(super_space)
        return space


    def init_weights(self):
        self.weights = {}
        self.weights["past_piano"] = stacked_rnn(self.recurrent_piano, "relu", self.dropout_probability)
        self.weights["future_piano"] = stacked_rnn(self.recurrent_piano, "relu", self.dropout_probability)
        self.weights["present_piano"] = MLP(self.mlp_piano, "relu", self.dropout_probability)
        self.weights["past_orchestra"] = stacked_rnn(self.recurrent_orch, "relu", self.dropout_probability)
        return
    
    def piano_embedding(self, piano_t, piano_past, piano_future):
        # Build input
        # Add a time axis to piano_t
        piano_t_time = tf.reshape(piano_t, [-1, 1, self.piano_dim])

        with tf.name_scope("present"):
            # Concatenate t and future
            x = piano_t
            for layer_ind, gru_layer in enumerate(self.weights["present_piano"]):
                with tf.name_scope("l_" + str(layer_ind)):
                    x = gru_layer(x)
                    keras_layer_summary(gru_layer)
            piano_t_emb = x

        with tf.name_scope("past"):
            # Concatenate t and future
            x = tf.concat([piano_past, piano_t_time], 1)
            for layer_ind, gru_layer in enumerate(self.weights["past_piano"]):
                with tf.name_scope("l_" + str(layer_ind)):
                    x = gru_layer(x)
                    keras_layer_summary(gru_layer)
            piano_past_emb = x

        with tf.name_scope("future"):
            # Concatenate t and future
            input_seq_fut = tf.concat([piano_t_time, piano_future], axis=1)
            # Flip the matrix along the time axis so that the last time index is t
            x = tf.reverse(input_seq_fut, axis=[1])
            for layer_ind, gru_layer in enumerate(self.weights["future_piano"]):
                with tf.name_scope("l_" + str(layer_ind)):
                    x = gru_layer(x)
                    keras_layer_summary(gru_layer)
            piano_future_emb = x
        
        return piano_t_emb, piano_past_emb, piano_future_emb

    def orchestra_embedding(self, orch_past):
        with tf.name_scope("past"):
            # Concatenate t and future
            x = orch_past
            for layer_ind, gru_layer in enumerate(self.weights["past_orchestra"]):
                with tf.name_scope("l_" + str(layer_ind)):
                    x = gru_layer(x)
                    keras_layer_summary(gru_layer)
        return x

    def predict(self, inputs_ph):

        piano_t, piano_past, piano_future, orch_past, _ = inputs_ph
        
        with tf.name_scope("piano_embedding"):
            piano_t_emb, piano_past_emb, piano_future_emb = self.piano_embedding(piano_t, piano_past, piano_future)

        with tf.name_scope("orchestra_embedding"):
            orchestra_emb = self.orchestra_embedding(orch_past)

        #####################
        # Concatenate and predict
        with tf.name_scope("top_level_prediction"):
            embedding_concat = tf.concat([orchestra_emb, piano_past_emb, piano_future_emb, piano_t_emb], axis=1)
            if self.dropout_probability > 0:
                top_input_drop = Dropout(self.dropout_probability)(embedding_concat)
            else:
                top_input_drop = embedding_concat
            dense_layer = Dense(self.orch_dim, activation='sigmoid', name='orch_pred')
            orch_prediction = dense_layer(top_input_drop)
            keras_layer_summary(dense_layer)
        #####################

        return orch_prediction, orch_prediction


