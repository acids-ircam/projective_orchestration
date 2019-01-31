#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import random
import collections

class Training_strategy(object):
	def __init__(self, num_k_folds=10, config_folder=None, database_path=None, logger=None):
		self.num_k_folds = num_k_folds
		self.config_folder = config_folder
		self.database_path = database_path
		self.logger = logger
		# Important for reproducibility
		self.random_inst = random.Random()
		self.random_inst.seed(1234)
		return

	def sort_and_shuffle_dict(self, dico):
		# Important to keep a consistent shuffling between trainings
		# Here sort dict, then shuffle them using random_inst seed
		od = collections.OrderedDict(sorted(dico.items()))
		kk =  list(od.keys())      # Python 3; use keys = d.keys() in Python 2
		self.random_inst.shuffle(kk)
		return kk