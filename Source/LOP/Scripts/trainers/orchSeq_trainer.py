#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Scripts.standard_learning.standard_trainer import Standard_trainer

import LOP.Utils.seq2seq_data_transformation as seq2seq_data_transformation

class orchSeq_trainer(Standard_trainer):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# Max number of note in one orchestral frame
		self.max_length_seq = kwargs["max_length_seq"]
		self.n_instru = kwargs["n_instru"]
		self.n_pc = 12
		self.n_octave=11
		# +1 is for start/end of chord token
		self.dim_orch_seq = 11+12+kwargs["n_instru"]+1
		return

	def build_variables_nodes(self, model, parameters):
		super().build_variables_nodes(model, parameters)

		# Nodes for computing loss when training
		# ONE-HOT VECTORS
		self.orch_t_instru_ph = tf.placeholder(tf.float32, shape=(None, self.max_length_seq, self.n_instru), name="orch_t_instru")
		self.orch_t_pc_ph = tf.placeholder(tf.float32, shape=(None, self.max_length_seq), name="orch_t_pc")
		self.orch_t_octave_ph = tf.placeholder(tf.float32, shape=(None, self.max_length_seq), name="orch_t_octave")

		# Nodes for teacher forcing during training
		self.decoder_input_ph = tf.placeholder(tf.int32, shape=(None, self.max_length_seq, self.dim_orch_seq), name="orch_t_instru")
		self.decoder_lengths_ph = tf.placeholder(tf.int32, shape=(None), name="orch_t_instru")	
		return 
		

	def build_preds_nodes(self, model)
		import pdb; pdb.set_trace()
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		self.preds_instru_train, self.preds_pc_train, self.preds_octave_train = model.predict_train(inputs_ph)
		self.preds_instru_train, self.preds_pc_train, self.preds_octave_train = model.predict_train(inputs_ph)
		return
	
	def build_distance(self, model, parameters):
		import pdb; pdb.set_trace()
		# Three terms
		distance_instru = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.orch_t_instru_ph, logits=self.preds_instru_train)
		distance_pc = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.orch_t_pc_ph, logits=self.preds_pc_train)
		distance_octave = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.orch_t_octave_ph, logits=self.preds_octave_train)
		distance = distance_instru + distance_pc + distance_octave

		temporal_mask = tf.maximum(self.temporal_mask_preds, self.temporal_mask_truth)
		masked_distance = distance * tf.to_float(temporal_mask)

		return masked_distance

	def build_loss_nodes(self, model, parameters):
		with tf.name_scope('loss'):
			with tf.name_scope('distance'):
				self.loss_val = self.build_distance(model, parameters)

			import pdb; pdb.set_trace()

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
	
	def save_nodes(self, model):
		tf.add_to_collection('loss', self.loss)
		tf.add_to_collection('loss_val', self.loss_val)
		tf.add_to_collection('train_step', self.train_step)
		tf.add_to_collection('keras_learning_phase', self.keras_learning_phase)
		tf.add_to_collection('inputs_ph', self.piano_t_ph)
		tf.add_to_collection('inputs_ph', self.piano_past_ph)
		tf.add_to_collection('inputs_ph', self.piano_future_ph)
		tf.add_to_collection('inputs_ph', self.orch_past_ph)
		tf.add_to_collection('inputs_ph', self.orch_future_ph)

		# inputs orch_t
		tf.add_to_collection('orch_t_instru_ph', self.orch_t_instru_ph)
		tf.add_to_collection('orch_t_pc_ph', self.orch_t_pc_ph)
		tf.add_to_collection('orch_t_octave_ph', self.orch_t_octave_ph)

		# preds
		tf.add_to_collection('preds_instru', self.preds_instru)
		tf.add_to_collection('preds_pc', self.preds_pc)
		tf.add_to_collection('preds_octave', self.preds_octave)
		
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
		self.loss = tf.get_collection("loss")[0]
		self.loss_val = tf.get_collection("loss_val")[0]
		self.sparse_loss_mean = tf.get_collection("sparse_loss_mean")[0]
		self.mask_orch_ph = tf.get_collection("mask_orch_ph")[0]
		self.train_step = tf.get_collection('train_step')[0]
		self.keras_learning_phase = tf.get_collection("keras_learning_phase")[0]
		return


	def training_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch, summarize_dict):

		import pdb; pdb.set_trace()

		feed_dict, _ = super().build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = True

		# Seq2seq nodes
		num_batch = orch.shape[0]
		orch_t_instru = np.zeros((num_batch, self.max_length_seq, self.n_instru))
		orch_t_pc = np.zeros((num_batch, self.max_length_seq, self.n_pc))
		orch_t_octave = np.zeros((num_batch, self.max_length_seq, self.n_octave))
		seq_lenghts = []
		for batch_ind in range(num_batch):
			orch_seq_instru, orch_seq_pc, orch_seq_octave = seq2seq_data_transformation.orch_t_to_seq(orch_t, mapping)
			this_seq_len = len(orch_seq_instru)
			orch_t_instru[batch_ind, :this_seq_len] = 
			orch_t_pc[batch_ind, :this_seq_len] = 
			orch_t_octave[batch_ind, :this_seq_len] = 
			seq_lenghts.append(this_seq_len)


		feed_dic[self.orch_t_instru_ph] = orch_t_instru
		feed_dic[self.orch_t_pc_ph] = orch_t_pc
		feed_dic[self.orch_t_octave_ph] = orch_t_octave
		feed_dict[self.decoder_input_ph] = np.concatenate([orch_t_instru, orch_t_pc, orch_t_octave], axis=-1)
		feed_dict[self.decoder_lengths_ph] = seq_lenghts

		
		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']

		if SUMMARIZE:
			_, loss_batch, preds_batch, sparse_loss_batch, summary = sess.run([self.train_step, self.loss, self.preds_instru_train, self.preds_pc_train, self.preds_octave_train, merged_node], feed_dict)
		else:
			_, loss_batch, preds_batch, sparse_loss_batch = sess.run([self.train_step, self.loss, self.preds_instru_train, self.preds_pc_train, self.preds_octave_train], feed_dict)
			summary = None

		debug_outputs = {
		}

		return loss_batch, preds_batch, debug_outputs, summary

	def valid_step(self, sess, batch_index, piano, orch, duration_piano, mask_orch):
		# Almost the same function as training_step here,  but in the case of NADE learning for instance, it might be ver different.
		feed_dict, orch_t = self.build_feed_dict(batch_index, piano, orch, duration_piano, mask_orch)
		feed_dict[self.keras_learning_phase] = False
		loss_batch, preds_batch = sess.run([self.loss_val, self.preds], feed_dict)

		return loss_batch, preds_batch, orch_t

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen, duration_piano_extracted):
		feed_dict, orch_t = self.build_feed_dict_long_range(t, piano_extracted, orch_extracted, orch_gen, duration_piano_extracted)
		feed_dict[self.keras_learning_phase] = False
		loss_batch, preds_batch = sess.run([self.loss_val, self.preds], feed_dict)
		return loss_batch, preds_batch, orch_t

	def generation_step(self, sess, batch_index, piano, orch_gen, duration_gen, mask_orch):
		# Exactly the same as the valid_step in the case of the standard_learner
		loss_batch, preds_batch, orch_t = self.valid_step(sess, batch_index, piano, orch_gen, duration_gen, mask_orch)
		return preds_batch