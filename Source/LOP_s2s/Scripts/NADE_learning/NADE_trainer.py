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
from LOP.Scripts.standard_learning.standard_trainer import Standard_trainer
from LOP.Utils.training_error import bin_Xent_NO_MEAN_tf

class NADE_trainer(Standard_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)		
		# Number of ordering used when bagging NADEs
		self.num_ordering = kwargs["num_ordering"]
		return

	def build_variables_nodes(self, model, parameters):
		super().build_variables_nodes(model, parameters)
		# These two nodes are for the NADE process
		# orch_t_ph will be the unmodified ground-truth
		# orch_pred indicates the mask orchestal frame used during NADE generative process
		# mask_input is the binary mask on the inputs
		self.mask_input = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="mask_input")		
		self.orch_pred = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="orch_pred")
		return

	def build_preds_nodes(self, model):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		self.preds, _ = model.predict(inputs_ph, self.orch_pred, self.mask_input)
		# At generation time
		self.context_embedding_in = tf.placeholder(tf.float32, shape=(None, model.context_embedding_size), name="context_embedding")
		self.context_embedding_out = model.embed_context(inputs_ph)
		self.preds_gen, _ = model.predict_knowing_context(self.context_embedding_in, self.orch_pred, self.mask_input)

		# if self.DEBUG["plot_nade_ordering_preds"]:
		# 	tf.summary.histogram('context_embedding', self.context_embedding_out, collections=['nade_prediction'])
		# 	tf.summary.histogram('orch_pred', self.orch_pred, collections=['nade_prediction'])

		# 	def image_sum(var, name):
		# 		shape_var=var.get_shape()
		# 		reshape_var = tf.reshape(var, [1, shape_var[0], shape_var[1], 1])
		# 		tf.summary.image(name, reshape_var, 1, collections=['nade_prediction'])
		# 		return
		# 	image_sum(self.context_embedding_out, "context_embedding_out")
		# 	image_sum(self.orch_pred, "orch_pred")
		return
	
	def build_loss_nodes(self, model, parameters):
		with tf.name_scope('loss'):
			self.loss, self.sparse_loss_mean = self.build_losses(model, self.preds, parameters, False)
		# Need to disinguish here, because preds_gen does not need the entire graph to be computed
		# Hence using perds here would require feeding more placeholders and doing useless extra computing
		with tf.name_scope('loss_val'):
			self.loss_val = self.build_losses(model, self.preds_gen, parameters, True)
		return

	def build_losses(self, model, preds, parameters, loss_val_bool):
		with tf.name_scope('distance'):
			distance = bin_Xent_NO_MEAN_tf(self.orch_t_ph, preds)

		# self.aaa = distance
	
		if model.sparsity_coeff != 0:
			with tf.name_scope('sparse_output_constraint'):
				sparse_loss, sparse_loss_mean = self.build_sparsity_term(model, parameters)
				loss_val_ = distance + sparse_loss
		else:
			loss_val_ = distance
			temp = tf.zeros_like(loss_val_)
			sparse_loss_mean = tf.reduce_mean(temp)

		with tf.name_scope("NADE_mask_input"):
			# Masked gradients are the values known in the input : so 1 - mask are used for gradient 
			loss_val_masked_ = (1-self.mask_input)*loss_val_
			# Mean along pitch axis
			loss_val_masked_mean = tf.reduce_mean(loss_val_masked_, axis=1)
			# NADE Normalization
			nombre_unit_masked_in = tf.reduce_sum(self.mask_input, axis=1)
			norm_nade = model.orch_dim / (model.orch_dim - nombre_unit_masked_in + 1)
			loss_val_masked = norm_nade * loss_val_masked_mean

		# Note: don't reduce_mean the validation loss, we need to have the per-sample value
		# if parameters['mask_orch']:
		# 	with tf.name_scope('mask_orch'):
		# 		self.loss_val = tf.where(mask_orch_ph==1, loss_val_masked, tf.zeros_like(loss_val_masked))
		# else:
		# 	self.loss_val = loss_val_masked
		loss_val = loss_val_masked
		if loss_val_bool:
			return loss_val
	
		mean_loss = tf.reduce_mean(loss_val)
		with tf.name_scope('weight_decay'):
			weight_decay = super().build_weight_decay(model)
			# Weight decay
			if model.weight_decay_coeff != 0:
				# Keras weight decay does not work...
				loss = mean_loss + weight_decay
			else:
				loss = mean_loss
		return loss, sparse_loss_mean

	def build_train_step_node(self, model, optimizer):
		super().build_train_step_node(model, optimizer)
		return
	
	def save_nodes(self, model):
		super().save_nodes(model)
		tf.add_to_collection('mask_input', self.mask_input)
		tf.add_to_collection('orch_pred', self.orch_pred)
		tf.add_to_collection('context_embedding_in', self.context_embedding_in)
		tf.add_to_collection('context_embedding_out', self.context_embedding_out)
		tf.add_to_collection('preds_gen', self.preds_gen)
		return

	def load_pretrained_model(self, path_to_model):
		# Restore model and preds graph
		super().load_pretrained_model(path_to_model)
		self.mask_input = tf.get_collection('mask_input')[0]
		self.orch_pred = tf.get_collection('orch_pred')[0]
		self.context_embedding_in = tf.get_collection('context_embedding_in')[0]
		self.context_embedding_out = tf.get_collection('context_embedding_out')[0]
		self.preds_gen  = tf.get_collection('preds_gen')[0]

		# No DEBUG when restoring a model for generation
		self.DEBUG['plot_nade_ordering_preds']=False
		self.DEBUG['save_accuracy_along_sampling']=False
		return

	def training_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch, summarize_dict):
		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = True
		
		# Generate a mask for the input
		batch_size, orch_dim = orch_t.shape
		mask = np.zeros_like(orch_t)
		for batch_ind in range(batch_size):
			# Number of known units
			d = random.randint(0, orch_dim)
			# Indices
			ind = np.random.random_integers(0, orch_dim-1, (d,))
			mask[batch_ind, ind] = 1

		#############################################
		#############################################
		#############################################
		# import pdb; pdb.set_trace()
		# # Compute test Jacobian, to check that gradients are set to zero : Test passed !
		# mask_deb = np.zeros_like(orch_t)
		# mask_deb[:,:20] = 1
		# feed_dict[self.mask_input] = mask_deb
		# feed_dict[self.orch_pred] = orch_t
		# for trainable_parameter in tf.trainable_variables():
		# 	if trainable_parameter.name == "dense_3/bias:0":
		# 		AAA = trainable_parameter
		# grads = tf.gradients(self.loss, AAA)
		# loss_batch, dydx = sess.run([self.loss, grads], feed_dict)
		#############################################
		#############################################
		#############################################
		
		feed_dict[self.mask_input] = mask
		# No need to mask orch_t here, its done in the tensorflow graph
		feed_dict[self.orch_pred] = orch_t

		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']
		
		if SUMMARIZE:
			_, loss_batch, preds_batch, sparse_loss_batch, summary = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean, merged_node], feed_dict)
		else:
			_, loss_batch, preds_batch, sparse_loss_batch = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean], feed_dict)
			summary = None

		debug_outputs = [sparse_loss_batch]

		return loss_batch, preds_batch, debug_outputs, summary

	def generate_mean_ordering(self, sess, feed_dict, orch_t):	
		batch_size, orch_dim = orch_t.shape

		orch_pred, loss_batch = self.orderless_NADE_generation(sess, feed_dict, orch_t)
		
		preds_mean_over_ordering = self.mean_parallel_prediction(batch_size, orch_pred)
		preds_sample_from_ordering = np.random.binomial(1, preds_mean_over_ordering)

		loss_batch_mean = self.mean_parallel_prediction(batch_size, loss_batch)

		return loss_batch_mean, preds_sample_from_ordering		

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
			
			# Update matrices
			for ordering_ind in range(self.num_ordering):
				batch_begin = batch_size * ordering_ind
				batch_end = batch_size * (ordering_ind+1)
				mask[batch_begin:batch_end, orderings[ordering_ind][d]] = 1
				##################################################
				# Do we sample or not ??????
				orch_pred[batch_begin:batch_end, orderings[ordering_ind][d]] = np.random.binomial(1, preds_batch[batch_begin:batch_end, orderings[ordering_ind][d]])
				##################################################

			##############################
			##############################
			# DEBUG
			# Observe the evolution of the accuracy along the sampling process
			if self.DEBUG["save_accuracy_along_sampling"]:
				accuracy_batch = np.mean(accuracy_measure(orch_t_reshape, orch_pred))
				accuracy_along_sampling.append(accuracy_batch)
			# Plot the predictions
			if self.DEBUG["plot_nade_ordering_preds"] and (self.DEBUG["batch_counter"]==(self.DEBUG["num_batch"]-1)):
				for ordering_ind in range(self.num_ordering):
					batch_begin = batch_size * ordering_ind
					batch_end = batch_size * (ordering_ind+1)
					np.save(self.DEBUG["plot_nade_ordering_preds"] + '/' + str(d) + '_' + str(ordering_ind) + '.npy', orch_pred[batch_begin:batch_end,:])
				mean_pred_batch = self.mean_parallel_prediction(batch_size, orch_pred)
				np.save(self.DEBUG["plot_nade_ordering_preds"] + '/' + str(d) + '_mean.npy', mean_pred_batch)
			##############################
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

	def mean_parallel_prediction(self, batch_size, matrix):
		# Mean over the different generations (Comb filter output)
		if len(matrix.shape) > 1:
			dim_1 = matrix.shape[1]
			mean_over_ordering = np.zeros((batch_size, dim_1))
		else:
			mean_over_ordering = np.zeros((batch_size,))
		ind_orderings = np.asarray([e*batch_size for e in range(self.num_ordering)])
		for ind_batch in range(batch_size):
			mean_over_ordering[ind_batch] = np.mean(matrix[ind_orderings], axis=0)
			ind_orderings += 1
		return mean_over_ordering

	def valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch):
		# Reduce number of validation samples to speed up valid step
		if len(batch_index) > 20:
			batch_index = batch_index[:20]

		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		loss_batch, preds_batch = self.generate_mean_ordering(sess, feed_dict, orch_t)
		
		return loss_batch, preds_batch, orch_t

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen, duration_piano):
		# This takes way too much time in the case of NADE, so just remove it
		feed_dict, orch_t = super().build_feed_dict_long_range(t, piano_extracted, orch_extracted, orch_gen, duration_piano)
		# loss_batch, preds_batch  = self.generate_mean_ordering(sess, feed_dict, orch_t)
		loss_batch = [0.]
		preds_batch = np.zeros_like(orch_t)
		return loss_batch, preds_batch, orch_t

	def generation_step(self, sess, batch_index, piano, orch_gen, duration_gen, mask_orch):
		# Exactly the same as the valid_step
		loss_batch, preds_batch, orch_t = self.valid_step(sess, batch_index, piano, orch_gen, duration_gen, mask_orch)
		return preds_batch