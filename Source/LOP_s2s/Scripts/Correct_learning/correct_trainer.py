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

class correct_trainer(Standard_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)		
		self.DEBUG = kwargs["debug"]
		return

	def build_variables_nodes(self, model, parameters):
		super().build_variables_nodes(model, parameters)
		# Index to be predicted
		self.pitch_mask_shape = model.pitch_mask_shape
		self.pitch_mask = tf.placeholder(tf.float32, shape=self.pitch_mask_shape, name="pitch_predicted")
		return

	def build_preds_nodes(self, model):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_t_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		self.preds = model.predict(inputs_ph, self.pitch_mask)
		return

	def build_loss_nodes(self, model):
		with tf.name_scope('distance'):
			distance = keras.losses.binary_crossentropy(self.orch_t_ph, self.preds)

		self.loss_val = distance 
	
		mean_loss = tf.reduce_mean(loss_val)
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
		super().save_nodes(model)
		tf.add_to_collection('pitch_mask', self.pitch_mask)
		return

	def load_pretrained_model(self, path_to_model):
		# Restore model and preds graph
		super().load_pretrained_model(path_to_model)
		self.pitch_mask = tf.get_collection('pitch_mask')[0]
		return

	def training_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch, summarize_dict):
		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)

		# Choose a pitch to predict (random (0, orch_dim))
		# To perform batch training we need choose one single pitch for all the batches
		import pdb; pdb.set_trace()
		index_pitch = random.randint(0, orch.shape[1]-1)
		pitch_mask = np.ones((self.pitch_mask_shape))
		pitch_mask[:, index_pitch] = 0
		feed_dict[self.pitch_mask] = pitch_mask
		feed_dict[self.keras_learning_phase] = True

		##############################
		# Here implement selection of pitch to balance between 0 and 1
		# Batch size de 1 ??? parce qu en batch ce sera impossible de choisir un pitch commun...
		##############################
		
		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']
		
		if SUMMARIZE:
			_, loss_batch, preds_batch, sparse_loss_batch, mask_pitch, summary = sess.run([self.train_step, self.loss, self.preds, self.mask_pitch, merged_node], feed_dict)
		else:
			_, loss_batch, preds_batch, sparse_loss_batch, mask_pitch = sess.run([self.train_step, self.loss, self.preds, self.mask_pitch], feed_dict)
			summary = None

		import pdb; pdb.set_trace()

		debug_outputs = [sparse_loss_batch]

		return loss_batch, preds_batch, debug_outputs, summary

	def valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch):
		# Sum prediction over all pitches
		orch_dim = orch.shape[1]
		feed_dict, orch_t = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = False
		preds_batch = np.zeros((len(batch_index), orch_dim))
		for pitch_predicted in range(orch_dim):		
			feed_dict[self.pitch_predicted_ph] = pitch_predicted
			loss_batch, this_pred_pitch = sess.run([self.loss_val, self.preds], feed_dict)
			preds_batch[:,pitch_predicted] = this_pred_pitch
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