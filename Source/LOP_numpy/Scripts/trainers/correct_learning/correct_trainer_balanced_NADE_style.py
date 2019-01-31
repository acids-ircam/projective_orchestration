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

from LOP.Utils.build_batch import build_batch

# FIRST VERSION:
# NADE style, mask input, concatenate mask

class correct_trainer_balanced_NADE_style(Standard_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)		
		self.DEBUG = kwargs["debug"]
		self.mean_iteration_per_note = kwargs["mean_iteration_per_note"]
		return

	def build_variables_nodes(self, model, parameters):
		self.piano_t_ph = tf.placeholder(tf.float32, shape=(None, model.piano_dim), name="piano_t")
		self.piano_past_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.piano_dim), name="piano_past")
		self.piano_future_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.piano_dim), name="piano_future")
		#
		self.orch_t_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="orch_t")
		self.orch_past_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.orch_dim), name="orch_past")
		self.orch_future_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.orch_dim), name="orch_past")
		# Orchestral mask
		self.mask_orch_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="mask_orch")
		# Index to be predicted
		self.pitch_mask = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="pitch_predicted")
		return

	def build_preds_nodes(self, model):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_t_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		self.preds, _ = model.predict(inputs_ph, self.pitch_mask)
		return

	def build_loss_nodes(self, model, parameters):
		with tf.name_scope('distance'):
			self.orch_t_masked = tf.boolean_mask(self.orch_t_ph, 1-self.pitch_mask)
			self.preds_masked = tf.boolean_mask(self.preds, 1-self.pitch_mask)
			# These are [batch, 1] size vectors
			distance = keras.losses.binary_crossentropy(self.orch_t_masked, self.preds_masked)

		self.loss_val = distance
	
		mean_loss = tf.reduce_mean(self.loss_val)
		with tf.name_scope('weight_decay'):
			weight_decay = super().build_weight_decay(model)
			# Weight decay
			if model.weight_decay_coeff != 0:
				# Keras weight decay does not work...
				self.loss = mean_loss + weight_decay
			else:
				self.loss = mean_loss
		return

	# def build_train_step_node(self, model, optimizer)
	
	def save_nodes(self, model):
		tf.add_to_collection('preds', self.preds)
		tf.add_to_collection('orch_t_ph', self.orch_t_ph)
		tf.add_to_collection('loss', self.loss)
		tf.add_to_collection('loss_val', self.loss_val)
		tf.add_to_collection('train_step', self.train_step)
		tf.add_to_collection('mask_orch_ph', self.mask_orch_ph)
		tf.add_to_collection('keras_learning_phase', self.keras_learning_phase)
		tf.add_to_collection('inputs_ph', self.piano_t_ph)
		tf.add_to_collection('inputs_ph', self.piano_past_ph)
		tf.add_to_collection('inputs_ph', self.piano_future_ph)
		tf.add_to_collection('inputs_ph', self.orch_past_ph)
		tf.add_to_collection('inputs_ph', self.orch_future_ph)
		tf.add_to_collection('pitch_mask', self.pitch_mask)
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
		self.loss_val = tf.get_collection("loss_val")[0]
		self.mask_orch_ph = tf.get_collection("mask_orch_ph")[0]
		self.train_step = tf.get_collection('train_step')[0]
		self.keras_learning_phase = tf.get_collection("keras_learning_phase")[0]
		self.pitch_mask = tf.get_collection('pitch_mask')[0]
		return

	def training_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch, summarize_dict):
		feed_dict = {}
		
		# Build batch
		piano_t, piano_past, piano_future, orch_past, orch_future, orch_t, mask_orch_t = build_batch(batch_index, piano, orch, duration_piano, mask_orch, len(batch_index), self.temporal_order)
		orch_dim = orch.shape[1]
		
		loss_batches = []

		feed_dict = {self.piano_t_ph: piano_t,
			self.piano_past_ph: piano_past,
			self.piano_future_ph: piano_future,
			self.orch_past_ph: orch_past,
			self.orch_future_ph: orch_future,
			self.orch_t_ph: orch_t,
			self.mask_orch_ph: mask_orch_t}
		feed_dict[self.keras_learning_phase] = True

		pitch_mask = np.ones((len(batch_index), orch_dim))

		for batch_ind in range(len(batch_index)):
			this_orch_t = orch_t[batch_ind]

			# Choose positive pitches and SAME NUMBER of negative pitches
			positive_indices = np.where(this_orch_t==1)[0]
			negative_indices = np.where(this_orch_t==0)[0]

			number_of_training_points = len(positive_indices)

			random.shuffle(negative_indices)
			negative_indices_kept = negative_indices[:number_of_training_points]
			training_points = list(negative_indices_kept) + list(positive_indices)

			for training_point in training_points:
				pitch_mask[batch_ind, training_point] = 0

		feed_dict[self.pitch_mask] = pitch_mask
			
		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']

		# Problem: we cannot do batch gradient anymore...
		if SUMMARIZE:
			_, loss_batch, preds_batch, summary = sess.run([self.train_step, self.loss, self.preds, merged_node], feed_dict)
		else:
			_, loss_batch, preds_batch, orch_t_masked, preds_masked = sess.run([self.train_step, self.loss, self.preds, self.orch_t_masked, self.preds_masked], feed_dict)
			summary = None
		
		debug_outputs = {"sparse_loss_batch": 0}
		
		return loss_batch, preds_batch, debug_outputs, summary

	def valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch):
		loss_batches = []
		# Sum prediction over all pitches
		orch_dim = orch.shape[1]
		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = False
		preds_batch = np.zeros((len(batch_index), orch_dim))
		index_pitches = list(range(orch_dim))
		random.shuffle(index_pitches)
		for index_pitch in index_pitches:		
			pitch_mask = np.ones((len(batch_index), orch_dim))
			pitch_mask[:, index_pitch] = 0
			feed_dict[self.pitch_mask] = pitch_mask
			loss_batch, preds = sess.run([self.loss_val, self.preds], feed_dict)
			# sample
			preds_batch[:, index_pitch] = preds[:, index_pitch]
			loss_batches.append(loss_batch)
		return loss_batches, preds_batch, orch_t

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen, duration_piano):
		# This takes way too much time in the case of NADE, so just remove it
		feed_dict, orch_t = super().build_feed_dict_long_range(t, piano_extracted, orch_extracted, orch_gen, duration_piano)
		# loss_batch, preds_batch  = self.generate_mean_ordering(sess, feed_dict, orch_t)
		loss_batch = [0.]
		preds_batch = np.zeros_like(orch_t)
		return loss_batch, preds_batch, orch_t

	def generation_step(self, sess, batch_index, piano, orch_gen, duration_gen, mask_orch):
		_, preds_batch, _ = self.valid_step(sess, batch_index, piano, orch_gen, duration_gen, mask_orch)
		return preds_batch