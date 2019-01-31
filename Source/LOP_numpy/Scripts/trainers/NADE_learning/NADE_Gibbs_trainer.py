#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import keras
import random
import time
import config
from keras import backend as K

from LOP.Utils.measure import accuracy_measure
from LOP.Scripts.NADE_learning.NADE_trainer import NADE_trainer
from LOP.Utils.training_error import bin_Xent_NO_MEAN_tf

class NADE_Gibbs_trainer(NADE_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		##############################
		# TEST
		self.gibbs_sampling_factor = 6
		##############################
		return

	# Only this mehthod has slightly changed
	def orderless_NADE_generation(self, sess, feed_dict, orch_t):
		# Pre-compute the context embedding 
		# which will be the same for all the orderings
		context_embedding = sess.run(self.context_embedding_out, feed_dict)
		
		# Generate the orderings in parallel -> duplicate the embedding and orch_t matrices along batch dim
		batch_size, orch_dim = orch_t.shape
		# Start with an orchestra prediction and mask equal to zero
		orch_t_reshape = np.concatenate([orch_t for _ in range(self.num_ordering)], axis=0)
		orch_pred = np.zeros_like(orch_t_reshape)
		mask = np.zeros_like(orch_t_reshape)
		context_embedding_reshape = np.concatenate([context_embedding for _ in range(self.num_ordering)], axis=0)
		
		# Three nodes to feed now: orch_pred, context_embedding, and mask
		feed_dict_known_context = {}
		feed_dict_known_context[self.orch_t_ph] = orch_t_reshape
		feed_dict_known_context[self.context_embedding_in] = context_embedding_reshape

		# Build the orderings (use the same ordering for all elems in batch)
		orderings = []
		for ordering_ind in range(self.num_ordering):
			# This ordering
			ordering = list(range(orch_dim))
			random.shuffle(ordering)
			orderings.append(ordering)

		if self.DEBUG["save_accuracy_along_sampling"]:
			accuracy_along_sampling = []

		# Loop over the length of the orderings
		for d in range(orch_dim):
			# Generate step
			feed_dict_known_context[self.orch_pred] = orch_pred
			feed_dict_known_context[self.mask_input] = mask
			
			loss_batch, preds_batch = sess.run([self.loss_val, self.preds_gen], feed_dict_known_context)
			
			##############################
			##############################
			# DEBUG
			# Observe the evolution of the accuracy along the sampling process
			if self.DEBUG["save_accuracy_along_sampling"]:
				accuracy_batch = np.mean(accuracy_measure(orch_t_reshape, preds_batch))
				accuracy_along_sampling.append(accuracy_batch)
			# Plot the predictions
			if self.DEBUG["plot_nade_ordering_preds"] and (self.DEBUG["batch_counter"]==(self.DEBUG["num_batch"]-1)):
				for ordering_ind in range(self.num_ordering):
					batch_begin = batch_size * ordering_ind
					batch_end = batch_size * (ordering_ind+1)
					np.save(self.DEBUG["plot_nade_ordering_preds"] + '/' + str(d) + '_' + str(ordering_ind) + '.npy', preds_batch[batch_begin:batch_end,:])
				mean_pred_batch = self.mean_parallel_prediction(batch_size, preds_batch)
				np.save(self.DEBUG["plot_nade_ordering_preds"] + '/' + str(d) + '_mean.npy', mean_pred_batch)
			##############################
			##############################
			
			# Update matrices
			for ordering_ind in range(self.num_ordering):
				batch_begin = batch_size * ordering_ind
				batch_end = batch_size * (ordering_ind+1)
				mask[batch_begin:batch_end, orderings[ordering_ind][d]] = 1
				##################################################
				# Mean-field or sampling ? Sampling because need binary values to move along the Gibbs/NADE process
				orch_pred[batch_begin:batch_end, orderings[ordering_ind][d]] = np.random.binomial(1, preds_batch[batch_begin:batch_end, orderings[ordering_ind][d]])
				##################################################

		# Now continue Gibbs sampling
		while(d < orch_dim*self.gibbs_sampling_factor):
			
			pitch_resampled = random.randint(0, orch_dim-1)
			d += 1
			
			##############################
			# Randomly set one value in the mask to zero
			mask[:, pitch_resampled] = 0
			##############################

			##############################
			feed_dict_known_context[self.orch_pred] = orch_pred
			feed_dict_known_context[self.mask_input] = mask
			##############################
			
			##############################
			loss_batch, preds_batch = sess.run([self.loss_val, self.preds_gen], feed_dict_known_context)
			##############################
			
			##############################
			##############################
			# DEBUG
			# Observe the evolution of the accuracy along the sampling process
			if self.DEBUG["save_accuracy_along_sampling"]:
				accuracy_batch = np.mean(accuracy_measure(orch_t_reshape, preds_batch))
				accuracy_along_sampling.append(accuracy_batch)
			# Plot the predictions
			if self.DEBUG["plot_nade_ordering_preds"] and (self.DEBUG["batch_counter"]==(self.DEBUG["num_batch"]-1)):
				for ordering_ind in range(self.num_ordering):
					batch_begin = batch_size * ordering_ind
					batch_end = batch_size * (ordering_ind+1)
					np.save(self.DEBUG["plot_nade_ordering_preds"] + '/' + str(d) + '_' + str(ordering_ind) + '.npy', preds_batch[batch_begin:batch_end,:])
				mean_pred_batch = self.mean_parallel_prediction(batch_size, preds_batch)
				np.save(self.DEBUG["plot_nade_ordering_preds"] + '/' + str(d) + '_mean.npy', mean_pred_batch)
			##############################
			##############################
			
			##############################
			# Write back the mask to 1	
			mask[:, pitch_resampled] = 1
			# Resample
			orch_pred[:, pitch_resampled] = np.random.binomial(1, preds_batch[:, pitch_resampled])
			##############################

		if self.DEBUG["plot_nade_ordering_preds"] and (self.DEBUG["batch_counter"]==(self.DEBUG["num_batch"]-1)):
			np.save(self.DEBUG["plot_nade_ordering_preds"] + '/truth.npy', orch_t)

		# Save accuracy_along_sampling
		if self.DEBUG["save_accuracy_along_sampling"]:
			save_file_path = self.DEBUG["save_accuracy_along_sampling"] + '/' + str(self.DEBUG["batch_counter"]) + '.txt'
			with open(save_file_path, 'w') as thefile:
				for item in accuracy_along_sampling:
	  				thefile.write("{:.4f}\n".format(100*item))

		return orch_pred, loss_batch