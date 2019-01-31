#!/usr/bin/env python
# -*- coding: utf8 -*-

# Boxplot 10 k-folds of different models
# Y-axis: accuracy
# X-axis: models

from matplotlib import pyplot as plt
import os
import csv
import numpy as np
import pickle as pkl
import collections

# (name, path)
models = {
	"LSTM": "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/recurrent_units/LSTM_plugged_base/LSTM",
	"GRU": "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/recurrent_units/LSTM_plugged_base/GRU",
	"RNN": "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/recurrent_units/LSTM_plugged_base/SimpleRNN",
}

data = collections.OrderedDict()
for model_name in ["RNN", "LSTM", "GRU"]:	
	model_path = models[model_name]
	csv_file = model_path + "/result_summary/accuracy.csv"
	these_points = []
	with open(csv_file, 'r') as ff:
		reader = csv.DictReader(ff, delimiter=';')
		for row in reader:
			these_points.append(-float(row["short_range_sampled"]))
	data[model_name] = these_points
plt.boxplot(data.values(), labels=data.keys())
plt.xlabel('Recurent unit type')
plt.ylabel('Accuracy')
plt.title('Performances of different recurrent units (10-fold evaluation)')
plt.show()

# data_loss = {}
# data_acc = {}
# for folder in folders:
# 	configs = os.listdir(folder)
# 	for config in configs:
# 		if config == ".DS_Store":
# 			continue
# 		model_params = pkl.load(open(os.path.join(folder, config, "model_params.pkl"), 'rb'))
# 		param = model_params[param_name]
# 		this_data_loss = []
# 		this_data_acc = []
# 		for fold in range(10):
# 			path_result_file = os.path.join(folder, config, str(fold), "result.csv")
# 			# Read result.csv
# 			with open(path_result_file, "rb") as f:
# 				reader = csv.DictReader(f, delimiter=';')
# 				elem = reader.next()
# 				this_loss = float(elem["loss"])
# 				this_acc = float(elem["accuracy"])
# 				if this_acc == 0:
# 					continue
# 				this_data_loss.append(this_loss)
# 				this_data_acc.append(this_acc)
# 		if param in data_loss.keys():
# 			data_loss[param].extend(this_data_loss)
# 			data_acc[param].extend(this_data_acc)
# 		else:
# 			data_loss[param] = this_data_loss
# 			data_acc[param] = this_data_acc

# # Sort by keys values
# import collections
# ordered_data_loss = collections.OrderedDict(sorted(data_loss.items()))
# ordered_data_acc = collections.OrderedDict(sorted(data_acc.items()))
# plt.boxplot(ordered_data_acc.values(), labels=ordered_data_acc.keys())
# plt.savefig("test.pdf")

# plt.boxplot(ordered_data_loss.values(), labels=ordered_data_loss.keys())
# plt.savefig("test_.pdf")


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