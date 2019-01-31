#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers.recurrent import GRU
from keras.layers import Dense, Dropout

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

from LOP.Models.Utils.weight_summary import keras_layer_summary

class LSTM_plugged_residual(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)

        # Hidden layers architecture
        self.MLP_piano_emb = model_param['MLP_piano_emb']
        self.GRU_orch_emb = model_param['GRU_orch_emb']
        self.last_MLP = model_param['last_MLP']

        return

    @staticmethod
    def name():
        return "LSTM_plugged_residual"
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

    def init_weights(self):
        return

    def predict(self, inputs_ph):

        piano_t, _, _, orch_past, _ = inputs_ph

        #####################
        # GRU for modelling past orchestra
        # First layer
        if len(self.GRU_orch_emb) > 1:
            return_sequences = True
        else:
            return_sequences = False
        
        with tf.name_scope("orchestra_embedding"):
            gru_layer = GRU(self.GRU_orch_emb[0], return_sequences=return_sequences, input_shape=(self.temporal_order, self.orch_dim),
                    activation='relu', dropout=self.dropout_probability)
            x = gru_layer(orch_past)
            keras_layer_summary(gru_layer, collections=["weights"])
        
            if len(self.GRU_orch_emb) > 1:
                # Intermediates layers
                for layer_ind in range(1, len(self.GRU_orch_emb)):
                    # Last layer ?
                    if layer_ind == len(self.GRU_orch_emb)-1:
                        return_sequences = False
                    else:
                        return_sequences = True
                    with tf.name_scope("orch_rnn_" + str(layer_ind)):
                        gru_layer = GRU(self.GRU_orch_emb[layer_ind], return_sequences=return_sequences,
                                activation='relu', dropout=self.dropout_probability)
                        x = gru_layer(x)
                        keras_layer_summary(gru_layer, collections=["weights"])

            lstm_out = x
        #####################
        
        #####################
        # gru out and piano(t)
        with tf.name_scope("piano_embedding"):
            x = piano_t
            for num_unit in self.MLP_piano_emb:
                dense_layer = Dense(num_unit, activation='relu')  # fully-connected layer with 128 units and ReLU activation
                x = dense_layer(x)
                keras_layer_summary(dense_layer, collections=["weights"])
                x = Dropout(self.dropout_probability)(x)
            piano_embedding = x
        #####################

        #####################
        # Concatenate and predict
        with tf.name_scope("top_layer_prediction"):
            x = keras.layers.concatenate([lstm_out, piano_embedding], axis=1)
            for num_unit in self.last_MLP:
                dense_layer = Dense(num_unit, activation='relu')
                x = dense_layer(x)
                keras_layer_summary(dense_layer, collections=["weights"])
                x = Dropout(self.dropout_probability)(x)
            # Residual
            dense_layer = Dense(self.orch_dim, name='orch_pred')
            residue = dense_layer(x)
            keras_layer_summary(dense_layer, collections=["weights"])
            # Input
            orch_tm1 = orch_past[:,-1, :]
            # Sum
            # activations = (orch_tm1 - 0.5) + residue
            activations = orch_tm1 + residue
            # Activation
            orch_prediction = keras.activations.sigmoid(activations)
        #####################

        embedding_concat = x
        return orch_prediction, embedding_concat