#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

from LOP.Models.Utils.weight_summary import keras_layer_summary

class RBM(Model_lop):
    def __init__(self, model_param, dimensions):
        Model_lop.__init__(self, model_param, dimensions)
        # Hidden layers architecture
        self.n_hidden = model_param['n_hidden']
        self.Gibbs_steps = model_param["Gibbs_steps"]
        return

    @staticmethod
    def name():
        return "RBM"
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
        self.weights['W_orch'] = tf.get_variable("W_orch", (self.orch_dim, self.n_hidden), initializer=tf.random_normal_initializer(), dtype=tf.float64)
        self.weights['a_orch_stat'] = tf.get_variable("a_orch_stat", (self.orch_dim), initializer=tf.zeros_initializer(), dtype=tf.float64)
        self.weights['W_piano'] = tf.get_variable("W_piano", (self.piano_dim, self.n_hidden), initializer=tf.random_normal_initializer(), dtype=tf.float64)
        self.weights['a_piano_stat'] = tf.get_variable("a_piano_stat", (self.piano_dim), initializer=tf.zeros_initializer(), dtype=tf.float64)
        self.weights['b_stat'] = tf.get_variable("b_stat", (self.n_hidden), initializer=tf.zeros_initializer(), dtype=tf.float64)
        return

    def sampling_bernoulli(self, means):
        with tf.name_scope("sampling_bernoulli"):
            distrib=tf.distributions.Bernoulli(means, dtype=tf.float64)
            sample = distrib.sample()
            return sample

    def free_energy(self, v_orch, v_piano):
        with tf.name_scope("free_energy"):
            a_orch_reshape = tf.reshape(self.weights["a_orch_stat"], [-1, 1])
            bias_term_orch = tf.matmul(v_orch, a_orch_reshape)

            a_piano_reshape = tf.reshape(self.weights["a_piano_stat"], [-1, 1])
            bias_term_piano = tf.matmul(v_piano, a_piano_reshape)

            in_term = tf.matmul(v_orch, self.weights["W_orch"]) + tf.matmul(v_piano, self.weights["W_piano"]) + self.weights["b_stat"]
            log_term = tf.log(1 + tf.exp(in_term))
            W_term = tf.reduce_sum(log_term, axis=1)
            
            ret = - tf.reshape(bias_term_orch, [-1]) - tf.reshape(bias_term_piano, [-1]) - W_term
            return ret, in_term, log_term

    def prop_up(self, v_orch, v_piano):
        with tf.name_scope("prop_up"):
            # Get hidden
            activation = tf.matmul(v_orch, self.weights["W_orch"]) + tf.matmul(v_piano, self.weights["W_piano"]) + self.weights["b_stat"]
            value_mean = tf.sigmoid(activation)
            value_sampled = self.sampling_bernoulli(value_mean)
            return activation, value_mean, value_sampled

    def prop_down(self, h):
        with tf.name_scope("prop_down"):
            # Only orch, don't need piano
            activation_orch = tf.matmul(h, tf.transpose(self.weights["W_orch"])) + self.weights["a_orch_stat"]
            value_mean_orch = tf.sigmoid(activation_orch)
            value_sampled_orch = self.sampling_bernoulli(value_mean_orch)
            return activation_orch, value_mean_orch, value_sampled_orch,\

    def run_K_Gibbs_step(self, v_orch_init, piano_t, num_steps):
        v_orch = v_orch_init
        for step in range(num_steps):
            # _, h_mean, h_sampled = self.prop_up(v_orch, piano_t)
            # if (step == num_steps-1):
            #     # Last update can be made with sampled values
            #     _, v_orch_mean, v_orch_sampled = self.prop_down(h_mean)
            # else:
            #     _, v_orch_mean, v_orch_sampled = self.prop_down(h_sampled)
            _, _, h_sampled = self.prop_up(v_orch, piano_t)
            _, v_orch_mean, v_orch_sampled = self.prop_down(h_sampled)
            v_orch = v_orch_sampled
        return v_orch_mean, v_orch_sampled

    def build_train_and_generate_nodes(self, inputs_ph, orch_t):
        piano_t, _, _, orch_past, _ = inputs_ph
        
        # CD-1
        with tf.name_scope("CD_training"):
            _, _, h0_sampled = self.prop_up(orch_t, piano_t)
            _, _, v_orch_sampled_train = self.prop_down(h0_sampled)
            negative_particle = tf.stop_gradient(v_orch_sampled_train)
            pos_term, _, _ = self.free_energy(orch_t, piano_t)
            neg_term, ccc, ddd = self.free_energy(negative_particle, piano_t)
            cost = pos_term - neg_term

        # # CD-K for training
        # # Initialization
        # v_orch = orch_t
        # # v_orch = tf.zeros_like(orch_t)
        # v_orch_mean_train, v_orch_sampled_train = self.run_K_Gibbs_step(v_orch, piano_t, self.Gibbs_steps)
        # tf.stop_gradient(v_orch_sampled_train)
        # pos_term, _, _ = self.free_energy(orch_t, piano_t)
        # neg_term, ccc, ddd = self.free_energy(v_orch_sampled_train, piano_t)
        # cost_10 = pos_term - neg_term

        # CD-K for generation
        with tf.name_scope("CD_generation"):
            orch_tm1 = tf.squeeze(orch_past[:,-1,:])
            # distrib = tf.distributions.Bernoulli(probs=[0.5])
            # v_orch = distrib.sample(tf.shape(orch_tm1))
            v_orch = orch_tm1
            # Alternate Gibbs sampling
            v_orch_mean_gen, v_orch_sampled_gen = self.run_K_Gibbs_step(v_orch, piano_t, self.Gibbs_steps)
        
        return cost, v_orch_mean_gen, v_orch_sampled_gen