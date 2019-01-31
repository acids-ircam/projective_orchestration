#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Hyperopt
import numpy as np
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

class cRBM_N(Model_lop):
    def __init__(self, model_param, dimensions):
        Model_lop.__init__(self, model_param, dimensions)
        # Hidden layers architecture
        self.n_hidden = model_param['n_hidden']
        self.n_visible = self.orch_dim
        self.n_condition = self.orch_dim * (self.temporal_order-1)+ self.piano_dim
        self.Gibbs_steps = model_param["Gibbs_steps"]
        return

    @staticmethod
    def name():
        return "cRBM"
    @staticmethod
    def binarize_piano():
        return True
    @staticmethod
    def binarize_orchestra():
        return True
    @staticmethod
    def is_keras():
        return False
    @staticmethod
    def optimize():
        return True
    @staticmethod
    def trainer():
        return "energy_based"
    @staticmethod
    def get_hp_space():
        super_space = Model_lop.get_hp_space()

        space = {
            'n_hidden': hopt_wrapper.qloguniform_int('n_hidden', log(100), log(5000), 10),
        }

        space.update(super_space)
        return space

    def init_weights(self):
        self.weights = {}
        self.weights['W'] = np.random((self.n_visible, self.n_hidden), dtype=np.float64)
        self.weights['A'] = np.random((self.n_condition, self.n_visible), dtype=np.float64)
        self.weights['B'] = np.random((self.n_condition, self.n_hidden), dtype=np.float64)
        self.weights['a_stat'] = np.zeros((self.n_visible), dtype=np.float64)
        self.weights['b_stat'] = np.zeros((self.n_hidden), dtype=np.float64)
        return

    def sampling_bernoulli(self, means):
        with tf.name_scope("sampling_bernoulli"):
            distrib=tf.distributions.Bernoulli(means, dtype=tf.float64)
            sample = distrib.sample()
            return sample

    def compute_dynamic_biases(self, context):
        self.a_dyn = self.weights["a_stat"] + tf.matmul(context, self.weights["A"])
        self.b_dyn = self.weights["b_stat"] + tf.matmul(context, self.weights["B"])
        return

    def free_energy(self, v):
        with tf.name_scope("free_energy"): 
            bias_term = tf.reduce_sum(tf.multiply(v, self.a_dyn), axis=1)
            in_term = tf.matmul(v, self.weights["W"]) + self.b_dyn
            W_term = tf.reduce_sum(tf.log(1 + tf.exp(in_term)), axis=1)
            ret = - bias_term - W_term
            return ret

    def prop_up(self, v):
        with tf.name_scope("prop_up"):
            # Get hidden
            activation = tf.matmul(v, self.weights["W"]) + self.b_dyn
            
            # RELU
            # value_mean = tf.nn.relu(activation)
            # value_sampled = value_mean
            # SIGMOID
            value_mean = tf.sigmoid(activation)
            value_sampled = self.sampling_bernoulli(value_mean)

            return activation, value_mean, value_sampled

    def prop_down(self, h):
        with tf.name_scope("prop_down"):
            # DROPOUT ?
            if self.dropout_probability>0:
                h_dropped = tf.nn.dropout(h, 1-self.dropout_probability)
            else:
                h_dropped = h
            activation = tf.matmul(h_dropped, tf.transpose(self.weights["W"])) + self.a_dyn
            value_mean = tf.sigmoid(activation)
            value_sampled = self.sampling_bernoulli(value_mean)
            return activation, value_mean, value_sampled

    def run_K_Gibbs_step(self, v_orch_init, num_steps):
        v_orch = v_orch_init
        for step in range(num_steps):
            _, h_mean, h_sampled = self.prop_up(v_orch)
            h0 = h_sampled
            if (step == num_steps-1):
                # MEAN VALUE OR NOT ?
                # _, v_orch_mean, v_orch_sampled = self.prop_down(h_mean)
                _, v_orch_mean, v_orch_sampled = self.prop_down(h0)
            else:
                _, v_orch_mean, v_orch_sampled = self.prop_down(h0)
            # v_orch = v_orch_mean
            v_orch = v_orch_sampled
        return v_orch_mean, v_orch_sampled

    def build_train_and_generate_nodes(self, inputs_ph, orch_t):
        piano_t, _, _, orch_past, _ = inputs_ph
        
        # Compute dynamic biases
        with tf.name_scope("dynamic_biase"):
            orch_past_reshape = tf.reshape(orch_past, [-1, (self.temporal_order-1)*self.orch_dim])
            context = tf.concat([piano_t, orch_past_reshape], axis=1)
            self.compute_dynamic_biases(context)
        
        # CD-1
        with tf.name_scope("CD_training"):
            _, h_mean, h_sampled = self.prop_up(orch_t)
            h0 = h_sampled
            _, _, v_orch_sampled_train = self.prop_down(h0)
            negative_sample = tf.stop_gradient(v_orch_sampled_train)
            pos_term = self.free_energy(orch_t)
            neg_term = self.free_energy(negative_sample)
            cost = pos_term - neg_term

        # # CD-K for training
        # # Initialization
        # v_orch = orch_t
        # # v_orch = tf.zeros_like(orch_t)
        # v_orch_mean_train, v_orch_sampled_train = self.run_K_Gibbs_step(v_orch, piano_t, self.Gibbs_steps)
        # tf.stop_gradient(v_orch_sampled_train)
        # pos_term, _, _ = self.free_energy(orch_t)
        # neg_term, ccc, ddd = self.free_energy(v_orch_sampled_train)
        # cost = pos_term - neg_term

        # CD-K for generation
        with tf.name_scope("CD_generation"):
            orch_tm1 = tf.squeeze(orch_past[:,-1,:])
            # distrib = tf.distributions.Bernoulli(probs=[0.5])
            # v_orch = distrib.sample(tf.shape(orch_tm1))
            v_orch = orch_tm1
            # Alternate Gibbs sampling
            v_orch_mean_gen, v_orch_sampled_gen = self.run_K_Gibbs_step(v_orch, self.Gibbs_steps)

        # BOULANGER mean over the batch before the substraction !!!!! WTF ?? A tester
        return cost, v_orch_mean_gen, v_orch_sampled_gen