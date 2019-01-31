#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp

from LOP.Models.Utils.weight_summary import variable_summary

class FGcRBM(Model_lop):
    def __init__(self, model_param, dimensions):
        Model_lop.__init__(self, model_param, dimensions)
        # Hidden layers architecture
        self.n_h = model_param['n_hidden']
        self.n_v = self.orch_dim
        self.n_c = self.orch_dim * (self.temporal_order-1)
        self.n_l = self.piano_dim
        self.n_f = model_param["n_factor"]

        self.n_fv = model_param["n_factor"]
        self.n_fh = model_param["n_factor"]

        self.Gibbs_steps = model_param["Gibbs_steps"]
        return

    @staticmethod
    def name():
        return "FGcRBM"
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
            'n_factor': hopt_wrapper.qloguniform_int('n_factor', log(200), log(1000), 10),
        }

        space.update(super_space)
        return space

    def init_weights(self):
        self.weights = {}
        # Main matrices
        self.weights['V'] = tf.get_variable("V", (self.n_v, self.n_f), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        self.weights['H'] = tf.get_variable("H", (self.n_h, self.n_f), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        self.weights['L'] = tf.get_variable("L", (self.n_l, self.n_f), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        # Dynamic biases visible
        self.weights['AV'] = tf.get_variable("AV", (self.n_v, self.n_fv), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        self.weights['AC'] = tf.get_variable("AC", (self.n_c, self.n_fv), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        self.weights['AL'] = tf.get_variable("AL", (self.n_l, self.n_fv), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        # Dynamic biases hidden
        self.weights['BH'] = tf.get_variable("BH", (self.n_h, self.n_fh), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        self.weights['BC'] = tf.get_variable("BC", (self.n_c, self.n_fh), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        self.weights['BL'] = tf.get_variable("BL", (self.n_l, self.n_fh), initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float64)
        # Static biases
        self.weights['a_stat'] = tf.get_variable("a_stat", (self.n_v), initializer=tf.zeros_initializer(), dtype=tf.float64)
        self.weights['b_stat'] = tf.get_variable("b_stat", (self.n_h), initializer=tf.zeros_initializer(), dtype=tf.float64)

        for key, value in self.weights.items():
            variable_summary(value, collections=["weights"])

        return

    def sampling_bernoulli(self, means):
        with tf.name_scope("sampling_bernoulli"):
            distrib=tf.distributions.Bernoulli(means, dtype=tf.float64)
            sample = distrib.sample()
            return sample

    def compute_dynamic_biases(self, context, latent):
        factors_a = tf.multiply(tf.matmul(context, self.weights["AC"]), tf.matmul(latent, self.weights["AL"]))
        self.a_dyn = self.weights["a_stat"] + tf.matmul(factors_a, tf.transpose(self.weights["AV"]))
        factors_b = tf.multiply(tf.matmul(context, self.weights["BC"]), tf.matmul(latent, self.weights["BL"]))
        self.b_dyn = self.weights["b_stat"] + tf.matmul(factors_b, tf.transpose(self.weights["BH"]))
        return

    def free_energy(self, v, latent):
        with tf.name_scope("free_energy"):
            # Bias term
            bias_term = tf.reduce_sum(tf.multiply(v, self.a_dyn), axis=1)
            # Factor terms
            factors = tf.multiply(tf.matmul(v, self.weights["V"]), tf.matmul(latent, self.weights["L"]))
            in_term = tf.matmul(factors, tf.transpose(self.weights["H"])) + self.b_dyn
            W_term = tf.reduce_sum(tf.log(1 + tf.exp(in_term)), axis=1)
            ret = - bias_term - W_term
            return ret, factors, in_term

    def prop_up(self, v, latent):
        with tf.name_scope("prop_up"):
            # Get hidden
            factors = tf.multiply(tf.matmul(v, self.weights["V"]), tf.matmul(latent, self.weights["L"]))
            activation = tf.matmul(factors, tf.transpose(self.weights["H"])) + self.b_dyn
            value_mean = tf.sigmoid(activation)
            value_sampled = self.sampling_bernoulli(value_mean)
            return activation, value_mean, value_sampled

    def prop_down(self, h, latent):
        with tf.name_scope("prop_down"):
            factors = tf.multiply(tf.matmul(h, self.weights["H"]), tf.matmul(latent, self.weights["L"]))
            activation = tf.matmul(factors, tf.transpose(self.weights["V"])) + self.a_dyn
            value_mean = tf.sigmoid(activation)
            value_sampled = self.sampling_bernoulli(value_mean)
            return activation, value_mean, value_sampled

    def run_K_Gibbs_step(self, v_orch_init, latent, num_steps):
        v_orch = v_orch_init
        for step in range(num_steps):
            _, h_mean, h_sampled = self.prop_up(v_orch, latent)
            if (step == num_steps-1):
                # _, v_orch_mean, v_orch_sampled, _, _, _ = self.prop_down(h_mean, latent)
                _, v_orch_mean, v_orch_sampled = self.prop_down(h_sampled, latent)
            else:
                _, v_orch_mean, v_orch_sampled = self.prop_down(h_sampled, latent)
            # v_orch = v_orch_mean
            v_orch = v_orch_sampled
        return v_orch_mean, v_orch_sampled

    def build_train_and_generate_nodes(self, inputs_ph, orch_t):
        piano_t, _, _, orch_past, _ = inputs_ph
        
        with tf.name_scope("build_units"):
            orch_past_reshape = tf.reshape(orch_past, [-1, (self.temporal_order-1)*self.orch_dim])
            context = orch_past_reshape
            latent = piano_t

        # Compute dynamic biases
        with tf.name_scope("dynamic_biase"):
            self.compute_dynamic_biases(context, latent)
        
        # CD-1
        with tf.name_scope("CD_training"):
            _, h0_mean, h0_sampled = self.prop_up(orch_t, latent)
            _, _, v_orch_sampled_train = self.prop_down(h0_sampled, latent)
            negative_sample = tf.stop_gradient(v_orch_sampled_train)
            pos_term, aa, bb = self.free_energy(orch_t, latent)
            neg_term, cc, dd = self.free_energy(negative_sample, latent)
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
            v_orch_mean_gen, v_orch_sampled_gen = self.run_K_Gibbs_step(v_orch, latent, self.Gibbs_steps)

        # BOULANGER mean over the batch before the substraction!! Why the fuck ?? A tester
        return cost, v_orch_mean_gen, v_orch_sampled_gen, aa, bb, cc, dd