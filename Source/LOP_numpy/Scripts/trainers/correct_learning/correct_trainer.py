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

# FIRST VERSION:
# batch training: the sample is randomly chosen and is the same for all batches

class correct_trainer(Standard_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)		
		self.DEBUG = kwargs["debug"]
		return

	def build_variables_nodes(self, model, parameters):
		super().build_variables_nodes(model, parameters)
		# Index to be predicted
		self.pitch_mask = tf.placeholder(tf.float32, shape=(model.orch_dim), name="pitch_predicted")
		return

	def build_preds_nodes(self, model):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_t_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		self.preds, _ = model.predict(inputs_ph, self.pitch_mask)
		return

	def build_loss_nodes(self, model, parameters):
		with tf.name_scope('distance'):
			orch_t_masked = tf.boolean_mask(self.orch_t_ph, 1-self.pitch_mask, axis=1)
			# These are [batch, 1] size vectors
			distance = keras.losses.binary_crossentropy(orch_t_masked, self.preds)

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
		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = True
		
		# Sum prediction over all pitches
		orch_dim = orch.shape[1]
		preds_batch = np.zeros((len(batch_index), orch_dim))
		index_pitches = list(range(orch_dim))
		random.shuffle(index_pitches)

		loss_batches = []
		
		for index_pitch in index_pitches:
			# Choose a pitch to predict (random (0, orch_dim))
			# To perform batch training we need choose one single pitch for all the batches
			pitch_mask = np.ones((orch_dim))
			pitch_mask[index_pitch] = 0
			feed_dict[self.pitch_mask] = pitch_mask
			
			SUMMARIZE = summarize_dict['bool']
			merged_node = summarize_dict['merged_node']

			if SUMMARIZE:
				_, loss_batch, preds_batch, summary = sess.run([self.train_step, self.loss, self.preds, merged_node], feed_dict)
			else:
				_, loss_batch, preds_batch = sess.run([self.train_step, self.loss, self.preds], feed_dict)
				summary = None

			loss_batches.append(loss_batch)

			break
		
		debug_outputs = {"sparse_loss_batch": 0}

		loss_batch = np.mean(loss_batches)

		return loss_batch, preds_batch, debug_outputs, summary

	def valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch):
		# Sum prediction over all pitches
		orch_dim = orch.shape[1]
		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = False
		preds_batch = np.zeros((len(batch_index), orch_dim))
		index_pitches = list(range(orch_dim))
		random.shuffle(index_pitches)
		for index_pitch in index_pitches:		
			pitch_mask = np.ones((orch_dim))
			pitch_mask[index_pitch] = 0
			feed_dict[self.pitch_mask] = pitch_mask
			loss_batch, preds = sess.run([self.loss_val, self.preds], feed_dict)
			# sample
			preds_batch[:, index_pitch] = preds[:, 0]
		return loss_batch, preds_batch, orch_t

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen, duration_piano):
		# This takes way too much time in the case of NADE, so just remove it
		feed_dict, orch_t = super().build_feed_dict_long_range(t, piano_extracted, orch_extracted, orch_gen, duration_piano)
		# loss_batch, preds_batch  = self.generate_mean_ordering(sess, feed_dict, orch_t)
		loss_batch = [0.]
		preds_batch = np.zeros_like(orch_t)
		return loss_batch, preds_batch, orch_t

	def generation_step(self, sess, batch_index, piano, orch_gen, duration_gen, mask_orch):
		orch_dim = orch_gen.shape[1]
		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		# Choose a pitch
		pitch_predicted = random.randint(0, orch_dim)
		# Exactly the same as the valid_step
		feed_dict[self.pitch_predicted_ph] = pitch_predicted
		this_pred_pitch = sess.run([self.preds], feed_dict)
		preds_batch = orch_t
		preds_batch[:, pitch_predicted] = this_pred_pitch
		return preds_batch