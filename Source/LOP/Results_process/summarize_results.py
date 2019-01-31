#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Write results of configs in .txt files

@author: leo
"""

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
import re
import shutil
import glob
import csv
import numpy as np

def main(config_folder, measure_name, avoid_folds):
    """Plots learning curves
    
    Paramaters
    ----------
    config_folders : str list
        list of strings containing the configurations from which we want to collect results
    name_measure_A : str
        name of the first measure (has to match a .npy file in the corresponding configuration folder)
    name_measure_B : str
        name of the second measure
        
    """

    plt.clf()

    res_summary = config_folder + '/result_summary'

    # csv file
    csv_file = os.path.join(res_summary, measure_name + '.csv')
    fieldnames = ['fold', 'short_range', 'short_range_sampled', 'long_range']
    with open(csv_file, 'w') as ff: 
        writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

    # Plot file
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    plot_file = os.path.join(res_summary, measure_name + '.csv')

    # List folds
    folds = glob.glob(config_folder + '/[0-9]*')

    folds = [e for e in folds if re.split("/", e)[-1] not in avoid_folds]

    # Best means
    best_mean_short = []
    best_mean_long = []

    # Collect numpy arrays
    for fold in folds:

        fold_num = re.split('/', fold)[-1]
        if "_" in fold_num:
            # This was a preprocessing folder
            continue
        else:
            fold_num = int(fold_num)

        ##############################
        # Summarize test results
        with open(fold + "/test_score.csv", 'r') as ff:
            reader = csv.DictReader(ff, delimiter=";")
            testRes_short_range = next(reader)
        with open(fold + "/test_score_sampled.csv", 'r') as ff:
            reader = csv.DictReader(ff, delimiter=";")
            testRes_sampled = next(reader)
        with open(fold + "/test_score_LR.csv", 'r') as ff:
            reader = csv.DictReader(ff, delimiter=";")
            testRes_long_range = next(reader)
        # Write csv
        with open(csv_file, 'a') as ff:
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")            
            writer.writerow({'fold': fold_num,
                'short_range': testRes_short_range[measure_name],
                'short_range_sampled': testRes_sampled[measure_name],
                'long_range' : testRes_long_range[measure_name]})
        ##############################

        ##############################
        # Plot validation errors along training
        results_long_range = np.loadtxt(fold + '/results_long_range/' + measure_name + '.txt')
        results_short_range = np.loadtxt(fold + '/results_short_range/' + measure_name + '.txt')

        with open(fold + '/results_long_range/' + measure_name + '_best_epoch.txt', 'r') as ff:
            long_range_best = int(ff.read())
        with open(fold + '/results_short_range/' + measure_name + '_best_epoch.txt', 'r') as ff:
            short_range_best = int(ff.read())

        plot_short, = plt.plot(results_short_range, color=colors[fold_num%len(colors)])
        plot_long, = plt.plot(results_long_range, color=colors[fold_num%len(colors)], ls='--', marker='o')

        if len(results_short_range.shape) > 0:
            best_mean_short.append(results_short_range[short_range_best])
            best_mean_long.append(results_long_range[long_range_best])
        else:
            best_mean_short = results_short_range
            best_mean_long = results_long_range

    # Legend and plots
    # traits = mlines.Line2D([], [], color='black',
    #                       markersize=15, label='Short range task')
    # ronds = mlines.Line2D([], [], color='black', marker='o',
    #                       markersize=15, label='Long range task')
    # plt.legend(handles=[traits, ronds])
    plt.legend([plot_short, plot_long], ['short range prediction', 'long range prediction'])
    plt.title(measure_name + ' curves')
    plt.xlabel('epochs')
    plt.ylabel(measure_name)
    plt.savefig(res_summary + '/' + measure_name + ".pdf")

    # Inter-measure csv file
    # Exists ?
    all_mes_file = res_summary + '/all_measures_foldMean.csv'
    fieldnames = ['measure', 'short term', 'long term']
    if not os.path.isfile(all_mes_file):
        with open(all_mes_file, 'w') as ff:
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
    with open(all_mes_file, 'a') as ff:
        writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
        writer.writerow({'measure': measure_name,
            'short term': np.mean(best_mean_short),
            'long term': np.mean(best_mean_long)})
    return

def summarize_several_configs(list_configs, save_folder):
    list_dict = []
    for config in list_configs:
        this_dict = {}
        # Read csv
        file_config = config + "/result_summary/" + "all_measures_foldMean.csv"
        if not os.path.isfile(file_config):
            continue
        with open(file_config, 'r') as ff:
            reader = csv.DictReader(ff, fieldnames =["measure", "short term", "long term"], delimiter=';')
            _ = next(reader)
            for line in reader:
                this_dict[line["measure"]] = float(line["short term"])
        this_dict["config"] = config
        list_dict.append(this_dict)
    
    # Sort 
    list_dict.sort(key=lambda x: x["accuracy"])

    # Write back in csv file
    fieldnames = list_dict[0].keys()
    out_file = save_folder + '/results_summary.csv'
    with open(out_file, 'w') as ff:
        writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for ll in list_dict:
            writer.writerow(ll)

if __name__ == '__main__':
    
    # root_folder ="/Users/leo/Recherche/lop/LOP/Experiments/Architecture/grid_searches/LSTM_plugged_base"
    # config_folders = glob.glob(root_folder + '/[0-9]*')
    # config_folders = glob.glob(root_folder + '/*')

    # root = "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/Add_more_information/temporal_order"
    # lll = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
    # config_folders = [root + "/" + e for e in lll]

    # config_folders = ["/Users/leo/Recherche/lop/LOP/Experiments/Architecture/Add_more_information/with_duration"]

    config_folders = [
    #     "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/nwx-ent_POS",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/Add_more_information/with_duration/norm_1",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/Add_more_information/with_duration/norm_2",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/Add_more_information/with_duration/norm_4",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/Add_more_information/with_duration/norm_128",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/grid_searches/Baseline_MLP_keras"
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_0",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_1",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_2",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_3",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_4",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_5",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_6",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_7",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_8",
        # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_9",
        # # "/Users/leo/Recherche/lop/LOP/Experiments/Error_function/Measure_results/w-acc_10",
        ]
    
    measures = ["accuracy", "loss", "precision", "recall"]

    avoid_folds = [] # Should be set to [] most of the times

    for config_folder in config_folders:

        if not os.path.isdir(config_folder):
            continue

        # Create store folder
        res_summary = config_folder + '/result_summary'
        if os.path.isdir(res_summary):
            shutil.rmtree(res_summary)
        os.mkdir(res_summary)

        for measure in measures:
            # try:
            #     main(config_folder, measure, avoid_folds)
            # except:
            #     print("Fail : " + config_folder + '\n' + measure)
            main(config_folder, measure, avoid_folds)

    # summarize_several_configs(config_folders, "/Users/leo/Recherche/lop/LOP/Experiments/Architecture/grid_searches/LSTM_plugged_base")
