#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.Future_past_piano.model_lop_future_past_piano import MLFPP

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

class Recurrent_embeddings(MLFPP):
    """Recurrent embeddings for both the piano and orchestral scores
    Piano embedding : p(t), ..., p(t+N) through stacked RNN. Last time index of last layer is taken as embedding.
    Orchestra embedding : o(t-N), ..., p(t) Same architecture than piano embedding.
    Then, the concatenation of both embeddings is passed through a MLP
    """
    def __init__(self, model_param, dimensions):

        MLFPP.__init__(self, model_param, dimensions)

        # Gru piano
        embeddings_size = model_param['embeddings_size']
        temp = model_param['hs_piano']
        self.hs_piano = list(temp)
        self.hs_piano.append(embeddings_size)

        temp = model_param['hs_orch']
        self.hs_orch = list(temp)
        self.hs_orch.append(embeddings_size)

        return

    @staticmethod
    def name():
        return (MLFPP.name() + "Recurrent_embeddings_0")
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
        super_space = MLFPP.get_hp_space()

        space = {
            'hs_piano': list_log_hopt(500, 2000, 10, 0, 2, 'hs_piano'),
            'hs_orch': list_log_hopt(500, 2000, 10, 0, 2, 'hs_orch'),
            'embeddings_size': qloguniform_int('hs_orch', log(500), log(1000), 10),
        }

        space.update(super_space)
        return space

    def piano_embedding(self, piano_t, piano_past, piano_future):
        # Build input
        # Add a time axis to piano_t
        piano_t_time = tf.reshape(piano_t, [-1, 1, self.piano_dim])

        with tf.name_scope("future"):
            # Concatenate t and future
            input_seq_fut = tf.concat([piano_t_time, piano_future], 1)
            # Flip the matrix along the time axis so that the last time index is t
            input_seq_fut = tf.reverse(input_seq_fut, [1])

            with tf.name_scope("stacked_gru"):
                piano_embedding_fut = stacked_rnn(input_seq_fut, self.hs_piano, 
                    rnn_type='gru', 
                    weight_decay_coeff=self.weight_decay_coeff, 
                    dropout_probability=self.dropout_probability, 
                    activation='relu'
                    )

        with tf.name_scope("past"):
            # Concatenate t and future
            input_seq_past = tf.concat([piano_past, piano_t_time], 1)
            with tf.name_scope("stacked_gru"):
                piano_embedding_past = stacked_rnn(input_seq_past, self.hs_piano, 
                    rnn_type='gru',
                    weight_decay_coeff=self.weight_decay_coeff,
                    dropout_probability=self.dropout_probability,
                    activation='relu'
                    )
        
        return piano_embedding_past, piano_embedding_fut

    def orchestra_embedding(self, orch_past):
        with tf.name_scope("gru"):
            orchestra_embedding = stacked_rnn(orch_past, self.hs_orch, 
                rnn_type='gru', 
                weight_decay_coeff=self.weight_decay_coeff, 
                dropout_probability=self.dropout_probability, 
                activation='relu'
                )       
        return orchestra_embedding

    def predict(self, inputs_ph):

        piano_t, piano_past, piano_future, orch_past, _ = inputs_ph
        
        with tf.name_scope("piano_embedding"):
            piano_embedding_past, piano_embedding_fut = self.piano_embedding(piano_t, piano_past, piano_future)

        with tf.name_scope("orchestra_embedding"):
            orchestra_embedding = self.orchestra_embedding(orch_past)

        #####################
        # Concatenate and predict
        with tf.name_scope("top_level_prediction"):
            with tf.name_scope("layer_0"):
                top_input = keras.layers.concatenate([orchestra_embedding, piano_embedding_past, piano_embedding_fut], axis=1)
                dense_layer = Dense(1000, activation='relu', name='orch_pred_0')
                top_0 = dense_layer(top_input)
                keras_layer_summary(dense_layer, collections=['weights'])
            with tf.name_scope("layer_1"):
                dense_layer = Dense(self.orch_dim, activation='sigmoid', name='orch_pred')
                orch_prediction = dense_layer(top_0)
                keras_layer_summary(dense_layer, collections=['weights'])
        #####################

        return orch_prediction, top_input


# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'hs_piano': [500],
# 'hs_orch': [600],
# 'embeddings_size': 500,
