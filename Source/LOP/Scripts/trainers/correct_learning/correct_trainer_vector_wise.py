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

class correct_trainer_vector_wise(Standard_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)		
		self.DEBUG = kwargs["debug"]
		return

	# def build_variables_nodes(self, model, parameters):

	# def build_preds_nodes(self, model):

	# def build_loss_nodes(self, model, parameters):

	# def build_train_step_node(self, model, optimizer)
	
	# def save_nodes(self, model):

	# def load_pretrained_model(self, path_to_model):
		
	# def training_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch, summarize_dict):

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