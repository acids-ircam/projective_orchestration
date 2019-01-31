#!/usr/bin/env python
# -*- coding: utf8 -*-

# Boxplot for a given parameter

from matplotlib import pyplot as plt
import glob
import os
import csv
import numpy as np
import pickle as pkl
import collections
import LOP.Scripts.config as config

param_name = "n_hidden"

fieldnames=["measure", "short term", "long term"]

# Collect the data
folders = "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/grid_searches/LSTM_plugged_base"
folders = glob.glob(folders + "/[0-9]*")

data_acc = {}

xx = []
yy = []
zz = []

for folder in folders:
	# Get config in fold 0
	model_params = pkl.load(open(os.path.join(folder + "/0/context/model_params.pkl"), 'rb'))
	param_value = model_params[param_name]

	# Get score in result_summary
	result_file = folder + "/result_summary/all_measures_foldMean.csv"
	if not os.path.isfile(result_file):
		print("No result file for " + folder)
		continue
	with open(result_file, 'r') as ff:
		reader = csv.DictReader(ff, fieldnames=fieldnames, delimiter=';')
		for elem in reader:
			if elem["measure"] == 'accuracy':
				this_data_acc = -float(elem["short term"])

	# if param_value in data_acc.keys():
	# 	data_acc[param_value].append(this_data_acc)
	# else:
	# 	data_acc[param_value] = [this_data_acc]

	# if (param_value==3) and (this_data_acc > 35):
	# 	import pdb; pdb.set_trace()

	if len(param_value) == 2:
		xx.append(param_value[0])
		yy.append(param_value[1])
		zz.append(this_data_acc)

zz_scale = [e/5 for e in zz]
plt.scatter(xx,yy,s=zz)
plt.savefig("GRU_n_hidden_2.pdf")

# # Sort by keys values
# ordered_data_acc = collections.OrderedDict(sorted(data_acc.items()))
# plt.boxplot(ordered_data_acc.values(), labels=ordered_data_acc.keys())
# plt.title("GRU: " + param_name)
# plt.savefig("GRU_" + param_name + ".pdf")


# def plot_dict(dict, title, xaxis, yaxis, filename):
# 	data = []

# 	for k, v in dict.iteritems():

# 		trace = go.Box(
# 			y=v,
# 			name="Fold_" + str(k),
# 		)

# 		data.append(trace)

# 	layout = {
# 		'title': title,
# 	    'xaxis': {
# 	        'title': xaxis,
# 	    },
# 	    'yaxis': {
# 	        'title': yaxis,
# 	    },
# 	    'boxmode': 'group',
# 	}

# 	fig = go.Figure(data=data, layout=layout)

# 	# Plot data
# 	py.plot(fig, filename=filename)

# plot_dict(loss_k_folds, "Difference of performances between folds (LSTM_based). Neg-ll", "fold index", "neg-ll", "k_folds_loss")
# plot_dict(acc_k_folds, "Difference of performances between folds (LSTM_based). Accuracy", "fold index", "neg-ll", "k_folds_acc")