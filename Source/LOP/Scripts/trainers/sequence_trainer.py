#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import keras
import time
from keras import backend as K

import LOP.Scripts.config as config
from LOP.Utils.training_error import accuracy_low_TN_tf, bin_Xent_tf, bin_Xen_weighted_0_tf, accuracy_tf, sparsity_penalty_l1, sparsity_penalty_l2, bin_Xen_weighted_1_tf, bin_Xen_weighted_Positive
from LOP.Utils.build_batch import build_batch_seq

class Sequence_trainer(object):
	
	def __init__(self, **kwargs):
		self.temporal_order = kwargs["temporal_order"]
		self.DEBUG = kwargs["debug"]
		self.tf_type = tf.float32
		self.np_type = np.float32
		return

	def build_variables_nodes(self, model, parameters):
		# Build nodes
		# Inputs
		self.piano_ph = tf.placeholder(self.tf_type, shape=(None, self.temporal_order, model.piano_dim), name="piano")
		self.orch_tm1 = tf.placeholder(self.tf_type, shape=(None, self.temporal_order, model.orch_dim), name="orch_tm1")
		#
		self.orch_ph = tf.placeholder(self.tf_type, shape=(None, self.temporal_order, model.orch_dim), name="orch")
		return

	def build_preds_nodes(self, model):
		# Prediction
		self.preds, self.embedding_concat = model.predict((self.piano_ph, self.orch_tm1))
		return
	
	def build_distance(self, model, parameters):
		distance = keras.losses.binary_crossentropy(self.orch_ph, self.preds)
		return distance
	
	def build_sparsity_term(self, model, parameters):
		# Add sparsity constraint on the output ? Is it still loss_val or just loss :/ ???
		sparsity_coeff = model.sparsity_coeff
		sparse_loss = sparsity_penalty_l1(self.preds)
		# sparse_loss = sparsity_penalty_l2(self.preds)
		
		# Try something like this ???
		# sparse_loss = case({tf.less(sparse_loss, 10): (lambda: tf.constant(0))}, default=(lambda: sparse_loss), exclusive=True)
		# sparse_loss = tf.keras.layers.LeakyReLU(tf.reduce_sum(self.preds, axis=1))
		
		sparse_loss = sparsity_coeff * sparse_loss
		# DEBUG purposes
		sparse_loss_mean = tf.reduce_mean(sparse_loss)
		return sparse_loss, sparse_loss_mean

	def build_weight_decay(self, model):
		weight_decay = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * model.weight_decay_coeff
		return weight_decay

	def build_loss_nodes(self, model, parameters):
		with tf.name_scope('loss'):
			with tf.name_scope('distance'):
				distance = self.build_distance(model, parameters)
				self.accuracy = accuracy_tf(self.orch_ph, self.preds)

			# Sparsity term
			if model.sparsity_coeff != 0:
				with tf.name_scope('sparse_output_constraint'):
					sparse_loss, sparse_loss_mean = self.build_sparsity_term(model, parameters)
					loss_val_ = distance + sparse_loss
					self.sparse_loss_mean = sparse_loss_mean
			else:
				loss_val_ = distance
				temp = tf.zeros_like(loss_val_)
				self.sparse_loss_mean = tf.reduce_mean(temp)

			self.loss_val = loss_val_
			
			# Mean the loss
			mean_loss = tf.reduce_mean(self.loss_val)

			# Weight decay
			if model.weight_decay_coeff != 0:
				with tf.name_scope('weight_decay'):
					weight_decay = Standard_trainer.build_weight_decay(self, model)
					# Keras weight decay does not work...
					self.loss = mean_loss + weight_decay
			else:
				self.loss = mean_loss
		return
		
	def build_train_step_node(self, model, optimizer):
		if model.optimize():
			# Some models don't need training
			gvs = optimizer.compute_gradients(self.loss)
			capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
			self.train_step = optimizer.apply_gradients(capped_gvs)
		else:
			self.train_step = None
		self.keras_learning_phase = K.learning_phase()
		return
	
	def save_nodes(self, model):
		tf.add_to_collection('preds', self.preds)
		tf.add_to_collection('loss', self.loss)
		tf.add_to_collection('loss_val', self.loss_val)
		tf.add_to_collection('sparse_loss_mean', self.sparse_loss_mean)
		tf.add_to_collection('train_step', self.train_step)
		tf.add_to_collection('keras_learning_phase', self.keras_learning_phase)
		tf.add_to_collection('inputs_ph', self.piano_ph)
		tf.add_to_collection('inputs_ph', self.orch_ph)
		if model.optimize():
			self.saver = tf.train.Saver()
		else:
			self.saver = None
		return

	def load_pretrained_model(self, path_to_model):
		# Restore model and preds graph
		self.saver = tf.train.import_meta_graph(path_to_model + '/model.meta')
		inputs_ph = tf.get_collection('inputs_ph')
		self.piano_ph, self.orch_ph = inputs_ph
		self.preds = tf.get_collection("preds")[0]
		self.loss = tf.get_collection("loss")[0]
		self.loss_val = tf.get_collection("loss_val")[0]
		self.sparse_loss_mean = tf.get_collection("sparse_loss_mean")[0]
		self.train_step = tf.get_collection('train_step')[0]
		self.keras_learning_phase = tf.get_collection("keras_learning_phase")[0]
		return

	def build_feed_dict(self, batch_index, piano, orch, duration_piano, mask_orch):
		# Build batch
		piano, orch_tm1, orch = build_batch_seq(batch_index, piano, orch, duration_piano, mask_orch, len(batch_index), self.temporal_order)
		# Train step
		feed_dict = {self.piano_ph: piano,
			self.orch_tm1_ph: orch_tm1,
			self.orch_ph: orch}
		return feed_dict

	def training_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch, summarize_dict):
		feed_dict = self.build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = True

		# import pdb; pdb.set_trace()
		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']

		if SUMMARIZE:
			_, loss_batch, preds_batch, sparse_loss_batch, summary = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean, merged_node], feed_dict)
		else:
			_, loss_batch, preds_batch, sparse_loss_batch = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean], feed_dict)
			summary = None

		debug_outputs = {
			"sparse_loss_batch": sparse_loss_batch,
		}

		return loss_batch, preds_batch, feed_dict[self.orch_ph], debug_outputs, summary

	def valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch):
		# Almost the same function as training_step here,  but in the case of NADE learning for instance, it might be ver different.
		feed_dict = self.build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = False
		loss_batch, preds_batch = sess.run([self.loss_val, self.preds], feed_dict)

		return loss_batch, preds_batch, orch

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen, duration_piano_extracted):
		return valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch)

	def generation_step(self, sess, batch_index, piano, orch_gen, duration_gen, mask_orch):
		# Exactly the same as the valid_step in the case of the standard_learner
		loss_batch, preds_batch, _ = self.valid_step(sess, batch_index, piano, orch_gen, duration_gen, mask_orch)
		return preds_batch