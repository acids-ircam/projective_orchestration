#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import keras
import time
import re
from keras import backend as K

import LOP.Scripts.config as config
from LOP.Utils.training_error import accuracy_low_TN_tf, bin_Xent_tf, bin_Xen_weighted_0_tf, accuracy_tf, sparsity_penalty_l1, sparsity_penalty_l2, bin_Xen_weighted_1_tf
from LOP.Utils.build_batch import build_batch

from LOP.Scripts.trainers.standard_trainer import Standard_trainer

class Energy_trainer(Standard_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.temporal_order = kwargs["temporal_order"]
		self.DEBUG = kwargs["debug"]
		self.tf_type = tf.float64
		self.np_type = np.float64
		return

	# def build_variables_nodes(self, model, parameters):

	def build_weight_decay(self, model):
		# Avoid biases
		weight_losses = [tf.nn.l2_loss(v) 
			for v in tf.trainable_variables() 
			if not re.search(r'_stat', v.name)]
		weight_decay = tf.add_n(weight_losses)
		return weight_decay

	# def build_train_step_node(self, model, optimizer)

	# def build_feed_dict(self, batch_index, piano, orch, duration_piano, mask_orch)

	def build_preds_nodes(self, model):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		# self.distance, self.preds_mean, self.preds, self.aa, self.bb, self.cc, self.dd = model.build_train_and_generate_nodes(inputs_ph, self.orch_t_ph)
		self.distance, self.preds_mean, self.preds = model.build_train_and_generate_nodes(inputs_ph, self.orch_t_ph)
		return

	def build_loss_nodes(self, model, parameters):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph)
		with tf.name_scope('loss'):
			# Mean the loss
			mean_loss = tf.reduce_mean(self.distance)
			# Weight decay
			if model.weight_decay_coeff != 0:
				with tf.name_scope('weight_decay'):
					weight_decay = self.build_weight_decay(model)
					self.loss = mean_loss + weight_decay * model.weight_decay_coeff
			else:
				self.loss = mean_loss
		return
	
	def save_nodes(self, model):
		tf.add_to_collection('preds', self.preds)
		tf.add_to_collection('orch_t_ph', self.orch_t_ph)
		tf.add_to_collection('loss', self.loss)
		tf.add_to_collection('mask_orch_ph', self.mask_orch_ph)
		tf.add_to_collection('train_step', self.train_step)
		tf.add_to_collection('keras_learning_phase', self.keras_learning_phase)
		tf.add_to_collection('inputs_ph', self.piano_t_ph)
		tf.add_to_collection('inputs_ph', self.piano_past_ph)
		tf.add_to_collection('inputs_ph', self.piano_future_ph)
		tf.add_to_collection('inputs_ph', self.orch_past_ph)
		tf.add_to_collection('inputs_ph', self.orch_future_ph)
		if model.optimize():
			self.saver = tf.train.Saver()
		else:
			self.saver = None
		return

	def load_pretrained_model(self, path_to_model):
		# Restore model and preds graph
		self.saver = tf.train.import_meta_graph(path_to_model + '/model.meta')
		inputs_ph = tf.get_collection('inputs_ph')
		self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph = inputs_ph
		self.orch_t_ph = tf.get_collection("orch_t_ph")[0]
		self.preds = tf.get_collection("preds")[0]
		self.loss = tf.get_collection("loss")[0]
		self.mask_orch_ph = tf.get_collection("mask_orch_ph")[0]
		self.train_step = tf.get_collection('train_step')[0]
		self.keras_learning_phase = tf.get_collection("keras_learning_phase")[0]
		return

	def build_feed_dict_long_range(self, t, piano_extracted, orch_extracted, orch_gen, duration_piano_extracted):
		if duration_piano_extracted is not None:
			dur_shape = duration_piano_extracted.shape
			dur_reshape = duration_piano_extracted.reshape([dur_shape[0], dur_shape[1], 1])
			piano_extracted = np.concatenate((piano_extracted, dur_reshape), axis=2)

		# We cannot use build_batch function here, but getting the matrices is quite easy
		piano_t = piano_extracted[:, t, :]
		piano_past = piano_extracted[:, t-(self.temporal_order-1):t, :]
		piano_future = piano_extracted[:, t+1:t+self.temporal_order, :]
		orch_t = orch_extracted[:, t, :]
		orch_past = orch_gen[:, t-(self.temporal_order-1):t, :]
		orch_future = orch_gen[:, t+1:t+self.temporal_order, :]
		mask_orch_t = np.ones_like(orch_t)
		
		# Train step
		feed_dict = {self.piano_t_ph: piano_t,
			self.piano_past_ph: piano_past,
			self.piano_future_ph: piano_future,
			self.orch_past_ph: orch_past,
			self.orch_future_ph: orch_future,
			self.orch_t_ph: orch_t,
			self.mask_orch_ph: mask_orch_t}
		return feed_dict, orch_t

	def training_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch, summarize_dict):
		feed_dict, _ = self.build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = True
		
		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']

		if SUMMARIZE:
			_, loss_batch, summary = sess.run([self.train_step, self.loss, merged_node], feed_dict)
		else:
			_, loss_batch = sess.run([self.train_step, self.loss], feed_dict)
			summary = None

		debug_outputs = {
			"sparse_loss_batch": 0,
		}

		preds_batch = np.zeros([len(batch_index), orch.shape[1]])

		return loss_batch, preds_batch, debug_outputs, summary

	def valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch):
		# Almost the same function as training_step here,  but in the case of NADE learning for instance, it might be ver different.
		feed_dict, orch_t = self.build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = False
		preds_batch = sess.run(self.preds, feed_dict)
		#############################################
		#############################################
		#############################################
		debug_outputs = {}
		# Compute test Jacobian, to check that gradients are set to zero : Test passed !
		if self.DEBUG["salience_embedding"]:
			orch_dim = orch.shape[1]
			dAcc_dEmbedding_N = tf.gradients(self.accuracy, self.embedding_concat)
			dPreds_dEmbedding_N = [tf.gradients(self.preds[:, orch_ind], self.embedding_concat) for orch_ind in range(orch_dim)]
			dAcc_dEmbedding, dPreds_dEmbedding = sess.run([dAcc_dEmbedding_N[0], dPreds_dEmbedding_N], feed_dict)
			for bind in range(len(batch_index)):
				dAcc_dEmbedding_N = tf.gradients(self.accuracy[bind], self.embedding_concat)
				dPreds_dEmbedding_N = [tf.gradients(self.preds[bind, orch_ind], self.embedding_concat) for orch_ind in range(orch_dim)]
				import pdb; pdb.set_trace()
				dAcc_dEmbedding, dPreds_dEmbedding = sess.run([dAcc_dEmbedding_N[0], dPreds_dEmbedding_N], feed_dict)

				dPreds_dEmbedding_STACK = np.stack([e[0] for e in dPreds_dEmbedding])

				np.save(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_pred.npy', preds_batch[bind])
				plt.imshow(preds_batch[bind], cmap='hot')
				plt.savefig(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_pred.pdf')
				#
				np.save(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_orch.npy', orch_t[bind])
				plt.imshow(orch_t[bind], cmap='hot')
				plt.savefig(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_orch.pdf')
				#
				np.save(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_dAcc_dEmbedding.npy', dAcc_dEmbedding)
				plt.imshow(dAcc_dEmbedding, cmap='hot')
				plt.savefig(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_dAcc_dEmbedding.pdf')
				#
				np.save(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_dPreds_dEmbedding.npy', dPreds_dEmbedding[bind])
				plt.imshow(dPreds_dEmbedding[bind], cmap='hot')
				plt.savefig(self.DEBUG["salience_embedding"] + '/' + str(bind) + '_dPreds_dEmbedding.pdf')
		#############################################
		#############################################
		#############################################
		
		loss_batch = np.zeros(len(batch_index))
		
		return loss_batch, preds_batch, orch_t

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen, duration_piano_extracted):
		feed_dict, orch_t = self.build_feed_dict_long_range(t, piano_extracted, orch_extracted, orch_gen, duration_piano_extracted)
		feed_dict[self.keras_learning_phase] = False
		preds_batch = sess.run(self.preds, feed_dict)
		
		num_batch = piano_extracted.shape[0]
		loss_batch = np.zeros((num_batch))

		return loss_batch, preds_batch, orch_t

	def generation_step(self, sess, batch_index, piano, orch_gen, duration_gen, mask_orch):
		# Exactly the same as the valid_step in the case of the standard_learner
		loss_batch, preds_batch, orch_t = self.valid_step(sess, batch_index, piano, orch_gen, duration_gen, mask_orch)
		return preds_batch