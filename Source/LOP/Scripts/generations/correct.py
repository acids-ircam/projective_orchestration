#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import random
from keras import backend as K


def correct(trainer, piano, silence_ind, duration_gen, path_to_config, model_name, orch_init, batch_size=5):
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

    # Restore model and preds graph
    tf.reset_default_graph() # First clear graph to avoid memory overflow when running training and generation in the same process
    trainer.load_pretrained_model(path_to_model)
    
    configSession = tf.ConfigProto()
    configSession.gpu_options.per_process_gpu_memory_fraction = 0.3

    # List of valid indices
    valid_indices = list(range(temporal_order, total_length-temporal_order))
    random.shuffle(valid_indices)

    # Iint orch
    orch_gen = orch_init

    with tf.Session(config=configSession) as sess:
            
        if is_keras:
            K.set_session(sess)

        trainer.saver.restore(sess, path_to_model + '/model')

        # for t in range(init_length, total_length):    A REMPLACER PAR TOTAL_LENGTH - TEMPORAL ORDER QUAND ON FERA AUSSI BACKAARD
        for t in valid_indices:
            # If piano is a silence, we automatically orchestrate by a silence (i.e. we do nothing)
            if t not in silence_ind:
                # Just duplicate the temporal index to create batch generation
                batch_index = np.tile(t, batch_size)

                prediction = trainer.generation_step(sess, batch_index, piano, orch_gen, duration_gen, None)

                # prediction should be a probability distribution. Then we can sample from it
                # Note that it doesn't need to be part of the graph since we don't use the sampled value to compute the backproped error
                prediction_sampled = np.random.binomial(1, prediction)
    
                orch_gen[:, t, :] = prediction_sampled
            else:
                prediction_sampled = np.zeros((batch_size, orch_dim))
                orch_gen[:, t, :] = prediction_sampled

    return orch_gen
