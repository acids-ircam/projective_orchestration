#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras import backend as K
import numpy as np
import time
import os
import pickle as pkl
import shutil
import LOP.Utils.measure as measure
from multiprocessing.pool import ThreadPool
# from multiprocessing.pool import Pool

import LOP.Scripts.config as config
import LOP.Utils.early_stopping as early_stopping
import LOP.Utils.model_statistics as model_statistics
from LOP.Scripts.asynchronous_load_mat import async_load_mat
import LOP.Scripts.training_utils as training_utils

from LOP.Scripts.import_functions.import_trainer import import_trainer

# Plot weights
import LOP.Results_process.plot_weights as plot_weights

from validate import validate

def train(model, train_splits_batches, valid_splits_batches, test_splits_batches, normalizer,
		  model_params, parameters, config_folder, start_time_train, logger_train):
	
	DEBUG = parameters["debug"]
	SUMMARIZE=DEBUG["summarize"]
	
	# Build DEBUG dict
	if DEBUG["save_measures"]:
		DEBUG["save_measures"] = config_folder+"/save_measures"

	# Time information used
	time_limit = parameters['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds

	# Reset graph before starting training
	tf.reset_default_graph()
		
	###### PETIT TEST VALIDATION
	# Use same validation and train set
	# piano_valid, orch_valid, valid_index = piano_train, orch_train, train_index	

	which_trainer = model.trainer()

	# Save it for generation. SO UGLY
	with open(os.path.join(config_folder, 'which_trainer'), 'w') as ff:
		ff.write(which_trainer)
	trainer = import_trainer(which_trainer, model_params, parameters)
	# Flag to know if the model has to be trained or not
	model_optimize = model.optimize()

	############################################################
	# Display informations about the models
	num_parameters = model_statistics.count_parameters(tf.get_default_graph())
	logger_train.info('** Num trainable parameters :  {}'.format(num_parameters))
	with open(os.path.join(config_folder, 'num_parameters.txt'), 'w') as ff:
		ff.write("{:d}".format(num_parameters))

	############################################################
	# Training
	logger_train.info("#" * 60)
	logger_train.info("#### Training")
	epoch = 0
	OVERFITTING = False
	TIME_LIMIT = False

	# Train error
	loss_tab = np.zeros(max(1, parameters['max_iter']))

	# Select criteria
	overfitting_measure = parameters["overfitting_measure"]
	save_measures = parameters['save_measures']

	# Short-term validation error
	valid_tabs = {
		'loss': np.zeros(max(1, parameters['max_iter'])),
		'accuracy': np.zeros(max(1, parameters['max_iter'])),
		'precision': np.zeros(max(1, parameters['max_iter'])),
		'recall': np.zeros(max(1, parameters['max_iter'])),
		'true_accuracy': np.zeros(max(1, parameters['max_iter'])),
		'f_score': np.zeros(max(1, parameters['max_iter'])),
		'Xent': np.zeros(max(1, parameters['max_iter']))
		}
	# Best epoch for each measure
	best_epoch = {
		'loss': 0, 
		'accuracy': 0, 
		'precision': 0, 
		'recall': 0, 
		'true_accuracy': 0, 
		'f_score': 0, 
		'Xent': 0
	}
	
	# Sampled preds measures
	valid_tabs_sampled = {
		'loss': np.zeros(max(1, parameters['max_iter'])),
		'accuracy': np.zeros(max(1, parameters['max_iter'])),
		'precision': np.zeros(max(1, parameters['max_iter'])),
		'recall': np.zeros(max(1, parameters['max_iter'])),
		'true_accuracy': np.zeros(max(1, parameters['max_iter'])),
		'f_score': np.zeros(max(1, parameters['max_iter'])),
		'Xent': np.zeros(max(1, parameters['max_iter']))
		}

	# Long-term validation error
	valid_tabs_LR = {
		'loss': np.zeros(max(1, parameters['max_iter'])), 
		'accuracy': np.zeros(max(1, parameters['max_iter'])), 
		'precision': np.zeros(max(1, parameters['max_iter'])), 
		'recall': np.zeros(max(1, parameters['max_iter'])), 
		'true_accuracy': np.zeros(max(1, parameters['max_iter'])), 
		'f_score': np.zeros(max(1, parameters['max_iter'])), 
		'Xent': np.zeros(max(1, parameters['max_iter']))
		}
	# Best epoch for each measure
	best_epoch_LR = {
		'loss': 0, 
		'accuracy': 0, 
		'precision': 0, 
		'recall': 0, 
		'true_accuracy': 0, 
		'f_score': 0, 
		'Xent': 0
	}

	### Timing file
	# open('timing', 'w').close()

	if parameters['memory_gpu']:
		# This apparently does not work
		configSession = tf.ConfigProto()
		configSession.gpu_options.per_process_gpu_memory_fraction = parameters['memory_gpu']
	else:
		configSession = None

	with tf.Session(config=configSession) as sess:

		# Only for models with shared weights
		model.init_weights()

		##############################
		# Create PH and nodes
		if parameters['pretrained_model'] is None:
			logger_train.info((u'#### Graph'))
			start_time_building_graph = time.time()
			trainer.build_variables_nodes(model, parameters)
			trainer.build_preds_nodes(model)
			trainer.build_loss_nodes(model, parameters)
			trainer.build_train_step_node(model, config.optimizer())
			trainer.save_nodes(model)
			time_building_graph = time.time() - start_time_building_graph
			logger_train.info("TTT : Building the graph took {0:.2f}s".format(time_building_graph))
		else:
			logger_train.info((u'#### Graph'))
			start_time_building_graph = time.time() 
			trainer.load_pretrained_model(parameters['pretrained_model'])
			time_building_graph = time.time() - start_time_building_graph
			logger_train.info("TTT : Loading pretrained model took {0:.2f}s".format(time_building_graph))
		
		if SUMMARIZE:
			tf.summary.scalar('loss', trainer.loss)
		##############################

		summarize_dict = {}

		if SUMMARIZE:
			merged_node = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter(config_folder + '/summary', sess.graph)
			train_writer.add_graph(tf.get_default_graph())
		else:
			merged_node = None
		summarize_dict['bool'] = SUMMARIZE
		summarize_dict['merged_node'] = merged_node

		if model.is_keras():
			K.set_session(sess)
		
		# Initialize weights
		if parameters['pretrained_model']: 
			trainer.saver.restore(sess, parameters['pretrained_model'] + '/model')
		else:
			sess.run(tf.global_variables_initializer())


		# if DEBUG:
		# 	sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		# 	sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
			
		N_matrix_files = len(train_splits_batches)
		
		#######################################
		# Load first matrix
		#######################################
		load_data_start = time.time()
		pool = ThreadPool(processes=1)
		async_train = pool.apply_async(async_load_mat, (normalizer, train_splits_batches[0]['chunks_folders'], parameters, trainer.np_type))
		matrices_from_thread = async_train.get()
		load_data_time = time.time() - load_data_start
		logger_train.info("Load the first matrix time : " + str(load_data_time))

		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		model_optimize = False
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################
		##############################################################################

		# For dumb baseline models like random or repeat which don't need training step optimization
		if model_optimize == False:
			# WARNING : first validation matrix is not necessarily the same as the first train matrix
			async_test = pool.apply_async(async_load_mat, (normalizer, test_splits_batches[0]['chunks_folders'], parameters, trainer.np_type))
			init_matrices_test = async_test.get()
			test_results, test_results_sampled, test_long_range_results, _, _ = validate(trainer, sess, 
					init_matrices_test, test_splits_batches, 
					normalizer, parameters,
					logger_train)
			training_utils.mean_and_store_results(test_results, valid_tabs, 0)
			training_utils.mean_and_store_results(test_results_sampled, valid_tabs_sampled, 0)
			training_utils.mean_and_store_results(test_long_range_results, valid_tabs_LR, 0)

			test_tab={}
			test_tab_sampled={}
			test_tab_LR={}
			training_utils.mean_and_store_results(test_results, test_tab, None)
			training_utils.mean_and_store_results(test_results_sampled, test_tab_sampled, None)
			training_utils.mean_and_store_results(test_long_range_results, test_tab_LR, None)

			accuracy_training = np.zeros((100))

			return accuracy_training, training_utils.remove_tail_training_curves(valid_tabs, 1), test_tab, best_epoch, \
				training_utils.remove_tail_training_curves(valid_tabs_sampled, 1), test_tab_sampled, best_epoch, \
				training_utils.remove_tail_training_curves(valid_tabs_LR, 1), test_tab_LR, best_epoch_LR

		accuracy_training = []

		# Training iteration
		while (not OVERFITTING and not TIME_LIMIT
			   and epoch != parameters['max_iter']):
			start_time_epoch = time.time()

			trainer.DEBUG['epoch'] = epoch

			train_cost_epoch = []
			sparse_loss_epoch = []
			
			train_time = time.time()

			this_batch_accuracy = []

			for file_ind_CURRENT in range(N_matrix_files):

				#######################################
				# Get indices and matrices to load
				#######################################
				# We train on the current matrix
				train_index = train_splits_batches[file_ind_CURRENT]['batches']
				# But load the one next one
				file_ind_NEXT = (file_ind_CURRENT+1) % N_matrix_files
				next_chunks = train_splits_batches[file_ind_NEXT]['chunks_folders']

				#######################################
				# Load matrix thread
				#######################################
				async_train = pool.apply_async(async_load_mat, (normalizer, next_chunks, parameters, trainer.np_type))
				piano_input, orch_transformed, duration_piano, mask_orch = matrices_from_thread
				
				#######################################
				# Train
				#######################################
				for batch_index in train_index:

					loss_batch, preds_batch, debug_outputs, summary = trainer.training_step(sess, batch_index, piano_input, orch_transformed, duration_piano, mask_orch, summarize_dict)

					orch_t = orch_transformed[batch_index]
					accuracy_batch = measure.accuracy_measure(orch_t, preds_batch)
					this_batch_accuracy.extend(accuracy_batch)

					# Keep track of cost
					train_cost_epoch.append(loss_batch)
					sparse_loss_batch = debug_outputs["sparse_loss_batch"]
					sparse_loss_epoch.append(sparse_loss_batch)

				#######################################
				# New matrices from thread
				#######################################
				del(matrices_from_thread)
				matrices_from_thread = async_train.get()

			this_batch_accuracy_mean = -100 * np.mean(this_batch_accuracy)
			accuracy_training.append(this_batch_accuracy_mean)
				
			train_time = time.time() - train_time
			logger_train.info("Training time : {}".format(train_time))

			### 
			# DEBUG
			if trainer.DEBUG["plot_weights"]:
				# weight_folder=config_folder+"/weights/"+str(epoch)
				weight_folder=config_folder+"/weights"
				plot_weights.plot_weights(sess, weight_folder)

			#
			###

			# WARNING : first validation matrix is not necessarily the same as the first train matrix
			# So now that it's here, parallelization is absolutely useless....
			async_valid = pool.apply_async(async_load_mat, (normalizer, valid_splits_batches[0]['chunks_folders'], parameters, trainer.np_type))

			if SUMMARIZE:
				if (epoch<5) or (epoch%10==0):
					# Note that summarize here only look at the variables after the last batch of the epoch
					# If you want to look at all the batches, include it in 
					train_writer.add_summary(summary, epoch)

			mean_loss = np.mean(train_cost_epoch)
			loss_tab[epoch] = mean_loss

			#######################################
			# Validate
			#######################################
			valid_time = time.time()
			init_matrices_validation = async_valid.get()

			# Create DEBUG folders
			if trainer.DEBUG["plot_nade_ordering_preds"]:
				AAA = config_folder+"/DEBUG/preds_nade/"+str(epoch)
				trainer.DEBUG["plot_nade_ordering_preds"] = AAA
				os.makedirs(AAA)
			if trainer.DEBUG["save_accuracy_along_sampling"]:
				AAA = config_folder+"/DEBUG/accuracy_along_sampling/"+str(epoch)
				trainer.DEBUG["save_accuracy_along_sampling"] = AAA
				os.makedirs(AAA)
			if trainer.DEBUG["salience_embedding"]:
				AAA = config_folder+"/DEBUG/salience_embeddings/"+str(epoch)
				trainer.DEBUG["salience_embedding"] = AAA
				os.makedirs(AAA)

			valid_results, valid_results_sampled, valid_long_range_results, preds_val, truth_val = \
				validate(trainer, sess, 
					init_matrices_validation, valid_splits_batches,
					normalizer, parameters,
					logger_train)
			valid_time = time.time() - valid_time
			logger_train.info("Validation time : {}".format(valid_time))

			training_utils.mean_and_store_results(valid_results, valid_tabs, epoch)
			training_utils.mean_and_store_results(valid_results_sampled, valid_tabs_sampled, epoch)
			training_utils.mean_and_store_results(valid_long_range_results, valid_tabs_LR, epoch)
			end_time_epoch = time.time()
			
			#######################################
			# Overfitting ? 
			if epoch >= parameters['min_number_iteration']:
				# Choose short/long range and the measure
				OVERFITTING = early_stopping.up_criterion(valid_tabs[overfitting_measure], epoch, parameters["number_strips"], parameters["validation_order"])
				if not OVERFITTING:
					# Also check for NaN
					OVERFITTING = early_stopping.check_for_nan(valid_tabs, save_measures, max_nan=3)
			#######################################

			#######################################
			# Monitor time (guillimin walltime)
			if (time.time() - start_time_train) > time_limit:
				TIME_LIMIT = True
			#######################################

			#######################################
			# Log training
			#######################################
			logger_train.info("############################################################")
			logger_train.info('Epoch : {} , Training loss : {} , Validation loss : {} \n \
		Validation accuracy : {:.3f} ; {:.3f} \n \
		Precision : {:.3f} ; {:.3f} \n \
		Recall : {:.3f} ; {:.3f} \n \
		Xent : {:.6f} ; {:.3f} \n \
		Sparse_loss : {:.3f}'
							  .format(epoch, mean_loss,
								valid_tabs['loss'][epoch],
								valid_tabs['accuracy'][epoch], valid_tabs_sampled['accuracy'][epoch],
								valid_tabs['precision'][epoch], valid_tabs_sampled['precision'][epoch],
								valid_tabs['recall'][epoch],  valid_tabs_sampled['recall'][epoch],
								valid_tabs['Xent'][epoch], valid_tabs_sampled['Xent'][epoch],
								np.mean(sparse_loss_epoch)))

			logger_train.info('Time : {}'.format(end_time_epoch - start_time_epoch))

			#######################################
			# Best model ?
			# Xent criterion
			start_time_saving = time.time()
			for measure_name, measure_curve in valid_tabs.items():
				best_measure_so_far = measure_curve[best_epoch[measure_name]]
				measure_for_this_epoch = measure_curve[epoch]
				if (measure_for_this_epoch <= best_measure_so_far) or (epoch==0):
					if measure_name in save_measures:
						trainer.saver.save(sess, config_folder + "/model_" + measure_name + "/model")
					best_epoch[measure_name] = epoch

				#######################################
				# DEBUG
				# Save numpy arrays of measures values
				if trainer.DEBUG["save_measures"]:
					if os.path.isdir(trainer.DEBUG["save_measures"]):
						shutil.rmtree(trainer.DEBUG["save_measures"])
					os.makedirs(trainer.DEBUG["save_measures"])
					for measure_name, measure_tab in valid_results.items():
						np.save(os.path.join(trainer.DEBUG["save_measures"], measure_name + '.npy'), measure_tab[:2000])
					np.save(os.path.join(trainer.DEBUG["save_measures"], 'preds.npy'), np.asarray(preds_val[:2000]))
					np.save(os.path.join(trainer.DEBUG["save_measures"], 'truth.npy'), np.asarray(truth_val[:2000]))
				#######################################
	   
			end_time_saving = time.time()
			logger_train.info('Saving time : {:.3f}'.format(end_time_saving-start_time_saving))
			#######################################

			if OVERFITTING:
				logger_train.info('OVERFITTING !!')

			if TIME_LIMIT:
				logger_train.info('TIME OUT !!')

			#######################################
			# Epoch +1
			#######################################
			epoch += 1

		#######################################
		# Test
		#######################################
		test_time = time.time()
		async_test = pool.apply_async(async_load_mat, (normalizer, test_splits_batches[0]['chunks_folders'], parameters, trainer.np_type))
		init_matrices_test = async_test.get()
		test_results, test_results_sampled, test_long_range_results, preds_test, truth_test = \
			validate(trainer, sess, 
				init_matrices_test, test_splits_batches,
				normalizer, parameters,
				logger_train)
		test_time = time.time() - test_time
		logger_train.info("Test time : {}".format(test_time))
		
		test_tab={}
		test_tab_sampled={}
		test_tab_LR={}
		training_utils.mean_and_store_results(test_results, test_tab, None)
		training_utils.mean_and_store_results(test_results_sampled, test_tab_sampled, None)
		training_utils.mean_and_store_results(test_long_range_results, test_tab_LR, None)

		logger_train.info("############################################################")
		logger_train.info("""## Test Scores
Loss : {}
Validation accuracy : {:.3f} %, precision : {:.3f} %, recall : {:.3f} %
True_accuracy : {:.3f} %, f_score : {:.3f} %, Xent : {:.6f}"""
		.format(
			test_tab['loss'], test_tab['accuracy'], test_tab['precision'],
			test_tab['recall'], test_tab['true_accuracy'], test_tab['f_score'], test_tab['Xent']))
		logger_train.info('Time : {}'
						  .format(test_time))


		#######################################
		# Close workers' pool
		#######################################
		pool.close()
		pool.join()

	return accuracy_training, training_utils.remove_tail_training_curves(valid_tabs, epoch), test_tab, best_epoch, \
		training_utils.remove_tail_training_curves(valid_tabs_sampled, epoch), test_tab_sampled, best_epoch,\
		training_utils.remove_tail_training_curves(valid_tabs_LR, epoch), test_tab_LR, best_epoch_LR

# bias=[v.eval() for v in tf.global_variables() if v.name == "top_layer_prediction/orch_pred/bias:0"][0]
# kernel=[v.eval() for v in tf.global_variables() if v.name == "top_layer_prediction/orch_pred/kernel:0"][0]