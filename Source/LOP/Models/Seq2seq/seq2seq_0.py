#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

import tensorflow as tf
import tf.contrib.rnn as tf_rnn

class Seq2seq_0(Model_lop):
	def __init__(self, model_param, dimensions):
		super().__init__(model_param, dimensions)
		self.n_instru = model_param["n_instru"]
		self.max_seq_length = model_param["max_seq_length"]
		return

	@staticmethod
	def name():
		return "Seq2seq_test"
	@staticmethod
	def binarize_piano():
		return True
	@staticmethod
	def binarize_orchestra():
		return True
	@staticmethod
	def is_keras():
		return False
	@staticmethod
	def optimize():
		return True
	@staticmethod
	def trainer():
		return "orchSeq_trainer"
	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()
		space = {}
		space.update(super_space)
		return space

	def init_weights(self):
		# Orch embedding
		num_units = [2000, 2000]
		cells = [tf_rnn.GRUCell(num_units=n) for n in num_units]
		self.orch_embedding_C = tf_rnn.MultiRNNCell(cells)

		# Piano embedding
		self.piano_embedding_C = tf.layers.Dense(2000, activation=tf.nn.relu, use_bias=True)

		# Decoder
		num_units = [2000, 2000]
		cells = [tf_rnn.GRUCell(num_units=n) for n in num_units]
		self.decoder_C = tf_rnn.MultiRNNCell(cells)
		return

	def split_pred(self, pred):
		# Split a prediction into its three components: instru, pc and octave
		instru = decoder_outputs[:, :, :self.n_instru]
		pc = decoder_outputs[:, :, self.n_instru:self.n_instru+12]
		octave = decoder_outputs[:, :, :self.n_instru+12:]
		return instru, pc, octave

	def concat_pred(self, instru, pc, octave):
		return tf.concat([instru, pc, octave])

	def encoder(self, piano_t, orch_past):
		#####################
		# GRU for modelling past orchestra
		with tf.name_scope("orch_embedding"):
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(self.orch_embedding_C, orch_past, time_major=False, dtype=tf.float32)
			# last hidden state as embedding
			orch_embedding = encoder_outputs[:, -1, :]
		#####################
		
		#####################
		# Embedding piano
		with tf.name_scope("piano_embedding"):
			# piano_embedding = tf_layers.stack(piano_t, tf_layers.fully_connected, [2000, 2000], activation_fn=tf.nn.relu)
			piano_embedding = self.piano_embedding_C(piano_t)
		#####################

		encoder_state = tf.concat([orch_embedding, piano_embedding], axis=1)

		return encoder_state

	def decoder_train(encoder_state, decoder_input, input_lengths):
		# decoder_input: target sentence, with <s> as first token
		# decoder_lenghts: [batch_size] int32 containing sequences length
		with tf.name_scope("decoder_train"):
			decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=self.decoder_C, input=decoder_input, initial_state=encoder_state, sequence_length=input_lengths, time_major=False, dtype=tf.float32)
		return decoder_outputs

	def predict_train(self, inputs_ph, decoder_input_ph, decoder_lengths_ph):
		piano_t, _, _, orch_past, _ = inputs_ph
		encoder_state = self.encoder(piano_t, orch_past)
		import pdb; pdb.set_trace()
		decoder_outputs = self.decoder_train(encoder_state, decoder_input_ph, decoder_lengths_ph)
		preds_instru, preds_pc, preds_octave = self.split_pred(decoder_outputs)
		return preds_instru, preds_pc, preds_octave

	def predict_infer(self, inputs_ph, init_token_ph):

		piano_t, _, _, orch_past, _ = inputs_ph

		encoder_state = self.encoder(piano_t, orch_past)

		# At time 0 use init token and encoder state
		pred_t = init_token_ph
		state_t = encoder_state
		for t in range(self.max_seq_length):
			# Generate one sample
			pred_t, state_t = self.decoder_C(inputs=pred_t, state=state_t)
			# Greedy sample
			preds_instru, preds_pc, preds_split_pred()



		decoder_outputs = self.decoder_train(encoder_state, decoder_input_ph, decoder_lengths_ph)
		# preds_instru, preds_pc, preds_octave = self.decoder(encoder_state, decoder_input_ph, decoder_lengths_ph)

		preds_instru = decoder_outputs[:, :, :self.n_instru]
		preds_pc = decoder_outputs[:, :, self.n_instru:self.n_instru+12]
		preds_octave = decoder_outputs[:, :, :self.n_instru+12:]

		return preds_instru, preds_pc, preds_octave