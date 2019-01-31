#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import re
import os
import LOP.Scripts.config as config

def load_matrices(chunk_path_list, parameters, np_type):
    """Input : 
        - chunk_path_list : list of splitted matrices to be concatenated

    This function build the matrix corresponing to a coherent ensemble of files, for example only train files
    """
    tt=0
    T_max = len(chunk_path_list)*parameters["chunk_size"] 
    orch_transformed = np.zeros((T_max, parameters["N_orchestra"]), dtype=np_type)
    if parameters["embedded_piano"]:
        piano_input = np.zeros((T_max, parameters["N_piano_embedded"]), dtype=np_type)
    else:
        piano_input = np.zeros((T_max, parameters["N_piano"]), dtype=np_type)

    if parameters["duration_piano"]:
        duration_piano = np.zeros((T_max,), dtype=np_type)

    if parameters["mask_orch"]:
        mask_orch = np.zeros((T_max, parameters["N_orchestra"]), dtype=np_type)
    
    for block_folder in chunk_path_list:
        piano_transformed_PART, piano_embedded_PART, orch_transformed_PART, duration_piano_PART, mask_orch_PART = load_matrix_NO_PROCESSING(block_folder, parameters["embedded_piano"], parameters["mask_orch"])
        
        length = piano_transformed_PART.shape[0]
        if parameters["embedded_piano"]:
            piano_input[tt:tt+length]=piano_embedded_PART
        else:
            piano_input[tt:tt+length]=piano_transformed_PART
        orch_transformed[tt:tt+length]=orch_transformed_PART
        if parameters["duration_piano"]:
            norm = 32.0
            duration_piano_PART_NORM = duration_piano_PART / norm
            duration_piano[tt:tt+length]=duration_piano_PART_NORM
        if parameters["mask_orch"]:
            mask_orch[tt:tt+length]=mask_orch_PART
        tt += length

    # Crop the last part (some chunks will be smaller than parameters["chunk_size"] )
    piano_input_cropped=piano_input[:tt]
    orch_transformed_cropped=orch_transformed[:tt]
    if parameters["mask_orch"]:
        mask_orch_cropped=mask_orch[:tt]
    else:
        mask_orch_cropped=None
    if parameters["duration_piano"]:
        duration_piano_cropped=duration_piano[:tt]
    else:
        duration_piano_cropped = None

    return piano_input_cropped, orch_transformed_cropped, duration_piano_cropped, mask_orch_cropped

def load_matrix_NO_PROCESSING(block_folder, embedded_piano_bool, mask_orch_bool):
    piano_file = os.path.join(config.data_root(), config.data_name(), block_folder, 'pr_piano.npy')
    orch_file = re.sub('piano', 'orch', piano_file)
    piano_embedded_file = re.sub('piano', 'piano_embedded', piano_file)
    duration_piano_file = re.sub('pr_piano', 'duration_piano', piano_file)

    pr_piano_transformed = np.load(piano_file)
    if embedded_piano_bool:
        pr_piano_embedded = np.load(piano_embedded_file)
    else:
        pr_piano_embedded = None
    pr_orch_transformed = np.load(orch_file)
    duration_piano = np.load(duration_piano_file)

    if mask_orch_bool:
        mask_orch_file = re.sub('piano', 'mask_orch', piano_file)
        mask_orch = np.load(mask_orch_file)
    else:
        mask_orch = None

    return pr_piano_transformed, pr_piano_embedded, pr_orch_transformed, duration_piano, mask_orch