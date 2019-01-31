#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import LOP.Database.build_data_aux as build_data_aux


def load_from_pair(tracks_path, quantization, binarize_piano, binarize_orch, temporal_granularity):
    ############################################################
    # Read piano midi file and orchestra score if defined
    ############################################################
    pr_piano, event_piano, duration_piano, _, name_piano, pr_orch, _, _, instru_orch, _, duration =\
        build_data_aux.process_folder(tracks_path, quantization, binarize_piano, binarize_orch, temporal_granularity)
    return pr_piano, event_piano, duration_piano, name_piano, pr_orch, instru_orch, duration

def load_solo(piano_midi, quantization, binarize_piano, temporal_granularity):
    # Read piano pr
    pr_piano = Read_midi(path, quantization).read_file()
    # Process pr_piano
    pr_piano = process_data_piano(pr_piano, binarize_piano)
    # Take event level representation
    if temporal_granularity == 'event_level':
        event_piano = get_event_ind_dict(pr_piano)
        pr_piano = warp_pr_aux(pr_piano, event_piano)
    else:
        event_piano = None

    name_piano = re.sub(r'/.*\.mid', '', piano_midi)

    duration = get_pianoroll_time(pr_piano)

    return pr_piano, event_piano, name_piano, None, None, duration


class Generator(object):
    def __init__(self, ):
        
    def load_config(self, config_folder):
        ########################
        # Load config and model
        parameters = pkl.load(open(config_folder + '/script_parameters.pkl', 'rb'))
        model_parameters = pkl.load(open(config_folder + '/model_params.pkl', 'rb'))
        # Set a minimum seed size, because for very short models you don't event see the beginning
        self.seed_size = max(model_parameters['temporal_order'], 10) - 1
        self.quantization = parameters['quantization']
        self.temporal_granularity = parameters['temporal_granularity']
        self.instru_mapping = parameters['instru_mapping']
        ########################

    def 
        