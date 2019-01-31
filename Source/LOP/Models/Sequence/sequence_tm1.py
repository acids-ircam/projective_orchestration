#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell

# Keras
import keras
from keras.layers import Dense, Dropout, Conv1D

import numpy as np

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

from LOP.Models.Utils.weight_summary import keras_layer_summary

class Sequence_tm1(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)
        # Hidden layers architecture
        self.n_hs = model_param['n_hidden'] + [int(self.orch_dim)]
        
        return

    @staticmethod
    def name():
        return "Sequence_tm1"
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

        piano_ph, orch_tm1_ph = inputs_ph

        import pdb; pdb.set_trace()

        #####################
        # GRU for modelling past orchestra
        # First layer
        gru_cells = [GRUCell(e, activation='relu') for e in self.n_hs]
        states = [tf.zeros([128, e]) for e in self.n_hs]

        # ON EST OBLIGE DUTILISER tf.nn.dynamic_rnn
        # https://stackoverflow.com/questions/42440565/how-to-feed-back-rnn-output-to-input-in-tensorflow
        # Juste passer des sequences de longueur 1 au moment du train

        for t in range(self.temporal_order):
            piano_t = piano_ph[:,t,:]
            # HERE IMPLEMENT TEACHER FORCING RELAXATION
            orch_tm1 = orch_tm1_ph[:,t,:]
            x = tf.concat([piano_t,orch_tm1], axis=1)
            new_states = []
            for gru_cell, state in zip(gru_cells, states):
                x, state_out = gru_cell(x, state)
                new_states.append(state_out)
            states = new_states
            orch_t.append(x)

        return orch_pred, orch_pred