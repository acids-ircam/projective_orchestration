#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import random
from keras import backend as K


def correct(trainer, piano, orch, silence_ind, duration_gen, path_to_config, model_name='model', orch_init=None, batch_size=5):
    # Perform N=batch_size orchestrations
    # Sample by sample generation
    # Input : 
    #   - piano score : numpy (time, pitch)
    #   - model
    #   - optionnaly the beginning of an orchestral score : numpy (time, pitch)
    # Output :
    #   - orchestration by the model    

    # Paths
    path_to_model = os.path.join(path_to_config, model_name)
    dimensions = pkl.load(open(path_to_config + '/../dimensions.pkl', 'rb'))
    is_keras = pkl.load(open(path_to_config + '/is_keras.pkl', 'rb'))

    # Get dimensions
    orch_dim = dimensions['orch_dim']
    temporal_order = dimensions['temporal_order']
    total_length = piano.shape[0]

    if orch_init is not None:
        init_length = orch_init.shape[0]
        assert (init_length < total_length), "Orchestration initialization is longer than the piano score"[0]
        assert (init_length + 1 >= temporal_order), "Orchestration initialization must be longer than the temporal order of the model"
    else:
        init_length = temporal_order - 1
        orch_init = np.zeros((init_length, orch_dim))

    # Instanciate generation
    if orch_gen is None:
        orch_gen = np.random.binomial(n=1, p=0.1, size=(batch_size, total_length, orch_dim))

    # Restore model and preds graph
    tf.reset_default_graph() # First clear graph to avoid memory overflow when running training and generation in the same process
    trainer.load_pretrained_model(path_to_model)

    with tf.Session() as sess:
            
        if is_keras:
            K.set_session(sess)

        trainer.saver.restore(sess, path_to_model + '/model')
        

        time_indices = list(range(init_length, total_length-temporal_order))
        pitch_indices = list(range(orch_dim))
        mean_iteration_per_note = 10
        # Gibbs sampling
        for iter in range(total_length*orch_dim*mean_iteration_per_note):
            silenced_time = True
            while silenced_time:
                time, pitch = (random.choice(time_indices), random.choice(pitch_indices))
                # If piano is a silence, we automatically orchestrate by a silence (i.e. we do nothing)
                silenced_time = (time in silence_ind)
            
                # Just duplicate the temporal index to create batch generation
                batch_index = np.tile(time, batch_size)

                ########
                ######## A ECRIRE
                # prediction = trainer.generation_step(sess, batch_index, piano, orch_gen, duration_gen, None)

                # # prediction should be a probability distribution. Then we can sample from it
                # # Note that it doesn't need to be part of the graph since we don't use the sampled value to compute the backproped error
                # prediction_sampled = np.random.binomial(1, prediction)
    
                orch_gen[:, time, pitch] = prediction_sampled
                ########
                ########

    return orch_gen