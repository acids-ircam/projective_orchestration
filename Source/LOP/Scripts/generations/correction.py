#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import logging
import pickle as pkl
import re
import numpy as np
import os
import time
import torch

from correct import correct
from LOP.Scripts.import_functions.import_trainer import import_trainer

import LOP.Database.build_data_aux as build_data_aux
import LOP.Scripts.generations.generation_utils as generation_utils
import LOP_database.utils.pianoroll_processing as pianoroll_processing


def generate_midi(config_folder_corr, score_source, save_folder, initialization_type, number_of_version, duration_gen, num_pass_correct, logger_generate):
    """This function generate the orchestration of a midi piano score
    
    Parameters
    ----------
    config_folder : str
        Absolute path to the configuration folder, i.e. the folder containing the saved model and the results
    score_source : str
        Either a path to a folder containing two midi files (piano and orchestration) or the path toa piano midi files
    number_of_version : int
        Number of version generated in a batch manner. Since the generation process involves sampling it might be interesting to generate several versions
    duration_gen : int
        Length of the generated score (in number of events). Useful for generating only the beginning of the piece.
    logger_generate : logger
        Instanciation of logging. Can be None
    """

    logger_generate.info("#############################################")
    logger_generate.info("Orchestrating : " + score_source)

    # Load parameters
    parameters = pkl.load(open(config_folder_corr + '/script_parameters.pkl', 'rb'))
    model_parameters_corr = pkl.load(open(config_folder_corr + '/model_params.pkl', 'rb'))
    
    seed_size = max(model_parameters_corr['temporal_order'], 10) - 1

    #######################
    # Load data
    if re.search(r'mid$', score_source):
        pr_piano, event_piano, duration_piano, name_piano, pr_orch, duration = generation_utils.load_solo(score_source, parameters["quantization"], parameters["binarize_piano"], parameters["temporal_granularity"])
    else:
        if initialization_type=="seed":
            pr_piano, event_piano, duration_piano, name_piano, pr_orch, duration = generation_utils.load_from_pair(score_source, parameters["quantization"], parameters["binarize_piano"], parameters["binarize_orch"], parameters["temporal_granularity"], align_bool=True)
        else:
            pr_piano, event_piano, duration_piano, name_piano, pr_orch, duration = generation_utils.load_from_pair(score_source, parameters["quantization"], parameters["binarize_piano"], parameters["binarize_orch"], parameters["temporal_granularity"], align_bool=False)

    if (duration is None) or (duration<duration_gen):
        logger_generate.info("Track too short to be used")
        return
    ########################

    ########################
    # Shorten
    # Keep only the beginning of the pieces (let's say a 100 events)
    pr_piano = pianoroll_processing.extract_pianoroll_part(pr_piano, 0, duration_gen)
    if parameters["duration_piano"]:
        duration_piano = np.asarray(duration_piano[:duration_gen])
    else:
        duration_piano = None
    if parameters["temporal_granularity"] == "event_level":
        event_piano = event_piano[:duration_gen]
    pr_orch = pianoroll_processing.extract_pianoroll_part(pr_orch, 0, duration_gen)
    ########################

    ########################
    # Instanciate piano pianoroll
    N_piano = parameters["instru_mapping"]['Piano']['index_max']
    pr_piano_gen = np.zeros((duration_gen, N_piano), dtype=np.float32)
    pr_piano_gen = build_data_aux.cast_small_pr_into_big_pr(pr_piano, 0, duration_gen, parameters["instru_mapping"], pr_piano_gen)
    pr_piano_gen_flat = pr_piano_gen.sum(axis=1)
    silence_piano = [e for e in range(duration_gen) if pr_piano_gen_flat[e]== 0]
    ########################

    ########################
    # Initialize orchestra pianoroll with orchestra seed (choose one)
    N_orchestra = parameters['N_orchestra']
    pr_orchestra_truth = np.zeros((duration_gen, N_orchestra), dtype=np.float32)
    pr_orchestra_truth = build_data_aux.cast_small_pr_into_big_pr(pr_orch, 0, duration_gen, parameters["instru_mapping"], pr_orchestra_truth)
    if initialization_type == "seed":
        pr_orchestra_seed = generation_utils.init_with_seed(pr_orch, number_of_version, duration_gen, N_orchestra, parameters["instru_mapping"])
    elif initialization_type == "zeros":
        pr_orchestra_seed = generation_utils.init_with_zeros(number_of_version, duration_gen, N_orchestra)
    elif initialization_type == "constant":
        const_value = 0.1
        pr_orchestra_seed = generation_utils.init_with_constant(number_of_version, duration_gen, N_orchestra, const_value)
    elif initialization_type == "random":
        proba_on = 0.01
        pr_orchestra_seed = generation_utils.init_with_random(number_of_version, duration_gen, N_orchestra, proba_on)
    ########################
    
    #######################################
    # Embed piano
    time_embedding = time.time()
    if parameters['embedded_piano']:
        # Load model
        embedding_path = parameters["embedding_path"]
        embedding_model = torch.load(embedding_path, map_location="cpu")

        # Build embedding (no need to batch here, len(pr_piano_gen) is sufficiently small)
        # Plus no CUDA here because : afradi of mix with TF  +  possibly very long piano chunks
        piano_resize_emb = np.zeros((len(pr_piano_gen), 1, 128)) # Embeddings accetp size 128 samples
        piano_resize_emb[:, 0, parameters["instru_mapping"]['Piano']['pitch_min']:parameters["instru_mapping"]['Piano']['pitch_max']] = pr_piano_gen
        piano_resize_emb_TT = torch.tensor(piano_resize_emb)
        piano_embedded_TT = embedding_model(piano_resize_emb_TT.float(), 0)
        pr_piano_gen_embedded = piano_embedded_TT.numpy()
    else:
        pr_piano_gen_embedded = pr_piano_gen
    time_embedding = time.time() - time_embedding
    #######################################

    ########################
    # Inputs' normalization
    normalizer = pkl.load(open(os.path.join(config_folder_corr, 'normalizer.pkl'), 'rb'))
    if parameters["embedded_piano"]:        # When using embedding, no normalization
        pr_piano_gen_norm = pr_piano_gen_embedded
    else:
        pr_piano_gen_norm = normalizer.transform(pr_piano_gen_embedded)
    ########################
    
    ########################
    # Store folder
    string = re.split(r'/', name_piano)[-1]
    name_track = re.sub('piano_solo.mid', '', string)
    generated_folder = save_folder + '/correction_' + initialization_type + '_init/' + name_track
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)
    ########################

    ########################
    # Get trainer
    with open(os.path.join(config_folder_corr, 'which_trainer'), 'r') as ff:
        which_trainer_corr = ff.read()
    # Trainer
    trainer_corr = import_trainer(which_trainer_corr, model_parameters_corr, parameters)
    ########################

    ############################################################
    # Generate
    ############################################################
    time_generate_0 = time.time()
    model_path = 'model_accuracy'
    pr_orchestra_gen = pr_orchestra_seed
    # Correction
    for pass_index in range(num_pass_correct):
        pr_orchestra_gen = correct(trainer_corr, pr_piano_gen_norm, silence_piano, duration_piano, config_folder_corr, model_path, pr_orchestra_gen, batch_size=number_of_version)
        prefix_name = 'corr_' + str(pass_index) + '_'
        generation_utils.reconstruct_generation(pr_orchestra_gen, event_piano, generated_folder, prefix_name, parameters, seed_size)

    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

    ############################################################
    # Reconstruct and write
    ############################################################
    prefix_name = 'final_'
    generation_utils.reconstruct_generation(pr_orchestra_gen, event_piano, generated_folder, prefix_name, parameters, seed_size)
    generation_utils.reconstruct_original(pr_piano_gen, pr_orchestra_truth, event_piano, generated_folder, parameters)
    return

if __name__ == '__main__':
    import LOP.Scripts.config as config
    
    save_folder = "/fast-1/leo/automatic_arrangement/Generations"

    # config_folder_corr = '/Users/leo/Recherche/lop/LOP/Results/Data_bp_bo_noEmb_tempGran32/LSTM_correction_vector_wise/TEST/0'
    
    # config_folder_corrs = ['/fast-1/leo/automatic_arrangement/Saved_models/LSTM_corr/' + str(e) for e in range (10)]
    config_folder_corrs = ['/fast-1/leo/automatic_arrangement/Saved_models/LSTM_corr/5']

    for config_folder_corr in config_folder_corrs:
        score_sources = []
        with open(config_folder_corr + '/test_names.txt', 'r') as ff:
            for row in ff:
                score_sources.append(config.database_root() + "/" + row.strip('\n'))
        logger = logging.getLogger('worker')
        logger.setLevel(logging.INFO)
        for initialization_type in ["zeros", "random", "constant"]:
            for score_source in score_sources:
                generate_midi(config_folder_corr, score_source, save_folder, initialization_type=initialization_type, number_of_version=3, duration_gen=100, num_pass_correct=5, logger_generate=logger)