#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Scripts.load_matrices import load_matrices
import time


def async_load_mat(normalizer, chunk_path_list, parameters, np_type):
    """Thread for loading matrices during training
    """ 
    # Load matrix
    piano_input, orch, duration_piano, mask_orch = load_matrices(chunk_path_list, parameters, np_type)
    # Normalization
    if not parameters["embedded_piano"]:
    	piano_input = normalizer.transform(piano_input)
    return piano_input, orch, duration_piano, mask_orch
