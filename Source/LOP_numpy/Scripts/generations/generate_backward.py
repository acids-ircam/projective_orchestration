#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras import backend as K


def generate_backward(trainer, piano, silence_ind, duration_gen, path_to_config, model_name='model', orch_init_tail=None, batch_size=5):
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

    init_length = orch_init_tail.shape[1]
    assert (init_length < total_length), "Orchestration initialization is longer than the piano score"[0]
    assert (init_length + 1 >= temporal_order), "Orchestration initialization must be longer than the temporal order of the model"

    # Instanciate generation
    orch_gen = np.zeros((batch_size, total_length+temporal_order, orch_dim))
    orch_gen[:, -init_length:, :] = orch_init_tail
    # Pad piano
    N_piano = piano.shape[1]
    piano_padded = np.concatenate([np.zeros([temporal_order, N_piano]), piano], axis=0)

    # Restore model and preds graph
    tf.reset_default_graph() # First clear graph to avoid memory overflow when running training and generation in the same process
    trainer.load_pretrained_model(path_to_model)
    
    configSession = tf.ConfigProto()
    configSession.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(config=configSession) as sess:
            
        if is_keras:
            K.set_session(sess)

        trainer.saver.restore(sess, path_to_model + '/model')

        # for t in range(init_length, total_length):    A REMPLACER PAR TOTAL_LENGTH - TEMPORAL ORDER QUAND ON FERA AUSSI BACKAARD
        for t in range(total_length + temporal_order - init_length, temporal_order, -1):
            # If piano is a silence, we automatically orchestrate by a silence (i.e. we do nothing)
            if t not in silence_ind:
                # Just duplicate the temporal index to create batch generation
                batch_index = np.tile(t, batch_size)

                prediction = trainer.generation_step(sess, batch_index, piano_padded, orch_gen, duration_gen, None)

                # prediction should be a probability distribution. Then we can sample from it
                # Note that it doesn't need to be part of the graph since we don't use the sampled value to compute the backproped error
                prediction_sampled = np.random.binomial(1, prediction)
    
                orch_gen[:, t, :] = prediction_sampled
            else:
                prediction_sampled = np.zeros((batch_size, orch_dim))
                orch_gen[:, t, :] = prediction_sampled

    return orch_gen[:, temporal_order:]