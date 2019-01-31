#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.Future_piano.model_lop_future_piano import MLFP

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import Dense, Conv1D, TimeDistributed

# Hyperopt
from math import log

from LOP.Models.Utils.weight_summary import keras_layer_summary
from LOP.Models.Utils.stacked_rnn import stacked_rnn
from LOP.Utils.hopt_utils import list_log_hopt
from LOP.Utils.hopt_wrapper import qloguniform_int, quniform_int

class Conv_recurrent_embedding_1(MLFP):
    """Recurrent embeddings for both the piano and orchestral scores
    Piano embedding : p(t), ..., p(t+N) through a convulotional layer and a stacked RNN. Last time index of last layer is taken as embedding.
    Orchestra embedding : o(t-N), ..., p(t) Same architecture than piano embedding.
    Then, the concatenation of both embeddings is passed through a MLP
    """
    def __init__(self, model_param, dimensions):

        MLFP.__init__(self, model_param, dimensions)

        # Piano embedding
        self.kernel_size_piano = model_param["kernel_size_piano"] # only pitch_dim, add 1 for temporal conv 
        embeddings_size = model_param['embeddings_size']
        temp = model_param['hs_piano']
        self.hs_piano = list(temp)
        self.hs_piano.append(embeddings_size)

        # Orchestra embedding
        self.kernel_size_orch = model_param["kernel_size_orch"] # only pitch_dim, add 1 for temporal conv 
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
            'kernel_size_piano': quniform_int('kernel_size_piano', 4, 24, 1),
            'kernel_size_orch': quniform_int('kernel_size_orch', 4, 24, 1),
            'hs_piano': list_log_hopt(500, 2000, 10, 0, 2, 'hs_piano'),
            'hs_orch': list_log_hopt(500, 2000, 10, 0, 2, 'hs_orch'),
            'embeddings_size': qloguniform_int('hs_orch', log(500), log(1000), 10),
        }

        space.update(super_space)
        return space

    def __init_weights():
        self.weights = {}

        ###############################
        # Conv layer, shared between past, present and future
        self.weights["piano_embed_conv"] = Conv1D(1, self.kernel_size_piano, activation='relu')
        ###############################

        ###############################
        # Past piano embedding
        if len(self.n_hs) > 1:
            return_sequences = True
        else:
            return_sequences = False
        self.weights["piano_past_emb_GRUs"] = [GRU(self.hs_piano[0], return_sequences=return_sequences, activation='relu', dropout=self.dropout_probability)]
        if len(self.n_hs) > 1:
            for layer_ind in range(1, len(self.hs_piano)):
                # Last layer ?
                if layer_ind == len(self.n_hs)-1:
                    return_sequences = False
                else:
                    return_sequences = True
                self.weights["piano_past_emb_GRUs"].append(GRU(self.n_hs[layer_ind], return_sequences=return_sequences, activation='relu', dropout=self.dropout_probability))
        ###############################

        ###############################
        # Future piano embedding
        if len(self.n_hs) > 1:
            return_sequences = True
        else:
            return_sequences = False
        self.weights["piano_future_emb_GRUs"] = [GRU(self.hs_piano[0], return_sequences=return_sequences, activation='relu', dropout=self.dropout_probability)]
        if len(self.n_hs) > 1:
            for layer_ind in range(1, len(self.hs_piano)):
                # Last layer ?
                if layer_ind == len(self.n_hs)-1:
                    return_sequences = False
                else:
                    return_sequences = True
                self.weights["piano_future_emb_GRUs"].append(GRU(self.n_hs[layer_ind], return_sequences=return_sequences, activation='relu', dropout=self.dropout_probability))
        ###############################

        ###############################
        # Conv layer for orchstra
        self.weights["orch_embed_conv"] = Conv1D(1, self.kernel_size_orch, activation='relu')
        ###############################

        return

    def piano_embedding(self, piano_t, piano_past, piano_future):
    
        # Shared conv layer        
        conv_layer = self.weights["piano_embed_conv"]
        conv_layer_Tdist = TimeDistributed(conv_layer, input_shape=(self.temporal_order-1, self.piano_dim, 1))

        ##############################
        # Past
        with tf.name_scope("embed_past"):
            with tf.name_scope("build_piano_input"):
                # Flip the matrix along the time axis so that the last time index is t
                input_seq = tf.reverse(piano_past, [1])
                # Format as a 4D
                input_seq = tf.reshape(input_seq, [-1, self.temporal_order-1, self.piano_dim, 1])

            with tf.name_scope("conv_piano"):
                xx = conv_layer_Tdist(input_seq)
                keras_layer_summary(conv_layer)
                # Remove the last useless dimension
                cropped_piano_dim = self.piano_dim-self.kernel_size_piano+1
                xx = tf.reshape(xx, [-1, self.temporal_order-1, cropped_piano_dim])

            with tf.name_scope("recurrent"):
                for rnn in self.weights["piano_past_emb_GRUs"]:
                    xx = rnn(xx)
            embed_past = xx
        ##############################

        ##############################
        # Future
        with tf.name_scope("embed_future"):
            with tf.name_scope("build_piano_input"):
                # Format as a 4D
                input_seq = tf.reshape(piano_future, [-1, self.temporal_order-1, self.piano_dim, 1])

            with tf.name_scope("conv_piano"):
                xx = conv_layer_Tdist(input_seq)
                keras_layer_summary(conv_layer)
                # Remove the last useless dimension
                cropped_piano_dim = self.piano_dim-self.kernel_size_piano+1
                xx = tf.reshape(xx, [-1, self.temporal_order-1, cropped_piano_dim])

            with tf.name_scope("recurrent"):
                for rnn in self.weights["piano_future_emb_GRUs"]:
                    xx = rnn(xx)
            embed_future = xx
        ##############################

        ##############################        
        # Present
        with tf.name_scope("embed_present"):
            with tf.name_scope("conv_piano"):
                xx = conv_layer(piano_t)
            with tf.name_scope("MLP"):
                for dense in self.weights["piano_present_MLPs"]:
                    xx=dense(xx)
            embed_present = xx
        ##############################

        piano_embedding = tf.concat([embed_past, embed_present, embed_future], axis=1)
        
        return piano_embedding

    def orchestra_embedding(self, orch_past):
        conv_layer = self.weights["orch_embed_conv"]
        conv_layer_timeDist = TimeDistributed(conv_layer, input_shape=(self.temporal_order-1, self.orch_dim, 1))
        with tf.name_scope("build_orch_input"):
            # Format as a 4D
            input_seq = tf.reshape(orch_past, [-1, self.temporal_order-1, self.orch_dim, 1])

        with tf.name_scope("conv_orch"):
            o0 = conv_layer_timeDist(input_seq)
            keras_layer_summary(conv_layer)

        # Remove the last useless dimension
        cropped_orch_dim = self.orch_dim-self.kernel_size_orch+1
        o0 = tf.reshape(o0, [-1, self.temporal_order-1, cropped_orch_dim])
        
        with tf.name_scope("gru"):
            orchestra_embedding = stacked_rnn(o0, self.hs_orch, 
                rnn_type='gru', 
                weight_decay_coeff=self.weight_decay_coeff, 
                dropout_probability=self.dropout_probability, 
                activation='relu'
                )
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

        return orch_prediction, top_input


# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'hs_piano': [500],
# 'hs_orch': [600],
# 'embeddings_size': 500,
