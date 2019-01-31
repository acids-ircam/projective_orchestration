#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import logging
import pickle as pkl
import re
import numpy as np
import os
import time
import torch

from LOP.Scripts.import_functions.import_trainer import import_trainer

from LOP.Scripts.generations.generate import generate
from LOP.Scripts.generations.generate_backward import generate_backward

import LOP.Database.build_data_aux as build_data_aux
import LOP.Scripts.generations.generation_utils as generation_utils
import LOP_database.utils.pianoroll_processing as pianoroll_processing

from LOP_database.midi.write_midi import write_midi


def generate_midi(config_folder_fd, score_source, save_folder, initialization_type, number_of_version, duration_gen, logger_generate):
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
    parameters = pkl.load(open(config_folder_fd + '/script_parameters.pkl', 'rb'))
    model_parameters_fd = pkl.load(open(config_folder_fd + '/model_params.pkl', 'rb'))
    seed_size = max(model_parameters_fd['temporal_order'], 10) - 1

    #######################
    # Load data
    if re.search(r'mid$', score_source):
        pr_piano, event_piano, duration_piano, name_piano, pr_orch, duration = generation_utils.load_solo(score_source, parameters["quantization"], parameters["binarize_piano"], parameters["temporal_granularity"])
    else:
        if initialization_type=="seed":
            pr_piano, event_piano, duration_piano, name_piano, pr_orch, duration = generation_utils.load_from_pair(score_source, parameters["quantization"], parameters["binarize_piano"], parameters["binarize_orch"], parameters["temporal_granularity"], align_bool=True)
        else:
            pr_piano, event_piano, duration_piano, name_piano, pr_orch, _ = generation_utils.load_from_pair(score_source, parameters["quantization"], parameters["binarize_piano"], parameters["binarize_orch"], parameters["temporal_granularity"], align_bool=False)
            duration = len(pr_piano['Piano'])

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
    ########################

    ########################
    # Initialize orchestra pianoroll with orchestra seed (choose one)
    N_orchestra = parameters['N_orchestra']
    pr_orchestra_truth = np.zeros((duration_gen, N_orchestra), dtype=np.float32)
    pr_orchestra_truth = build_data_aux.cast_small_pr_into_big_pr(pr_orch, 0, duration_gen, parameters["instru_mapping"], pr_orchestra_truth)
    if initialization_type == "seed":
        pr_orchestra_seed = generation_utils.init_with_seed(pr_orch, number_of_version, seed_size, N_orchestra, parameters["instru_mapping"])
    elif initialization_type == "zeros":
        pr_orchestra_seed = generation_utils.init_with_zeros(number_of_version, seed_size, N_orchestra)
        # append zeros before piano to avoid zero init in orchestra
        pr_piano_gen = np.concatenate((np.zeros((seed_size, N_piano)), pr_piano_gen[:-seed_size]))
    elif initialization_type == "constant":
        const_value = 0.1
        pr_orchestra_seed = generation_utils.init_with_constant(number_of_version, seed_size, N_orchestra, const_value)
        pr_piano_gen = np.concatenate((np.zeros((seed_size, N_piano)), pr_piano_gen[:-seed_size]))
    elif initialization_type == "random":
        proba_activation = 0.1
        pr_orchestra_seed = generation_utils.init_with_random(number_of_version, seed_size, N_orchestra, proba_activation)
        pr_piano_gen = np.concatenate((np.zeros((seed_size, N_piano)), pr_piano_gen[:-seed_size]))
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
    # Detect silences
    pr_piano_gen_flat = pr_piano_gen.sum(axis=1)
    silence_piano = [e for e in range(duration_gen) if pr_piano_gen_flat[e]== 0]
    ########################

    ########################
    # Inputs' normalization
    normalizer = pkl.load(open(os.path.join(config_folder_fd, 'normalizer.pkl'), 'rb'))
    if parameters["embedded_piano"]:        # When using embedding, no normalization
        pr_piano_gen_norm = pr_piano_gen_embedded
    else:
        pr_piano_gen_norm = normalizer.transform(pr_piano_gen_embedded)
    ########################

    ########################
    # Create mask for instrumentation (only works for NADE)
    list_instru = ["Violin", "Viola", "Violoncello", "Contrabass"]
    instru_mapping = parameters["instru_mapping"]
    mask_instrumentation = np.zeros((N_orchestra))# 1 where instru are allowed, 0 else
    for instru in list_instru:
        index_min = instru_mapping[instru]["index_min"]
        index_max = instru_mapping[instru]["index_max"]
        mask_instrumentation[index_min:index_max] = 1
    ########################
    
    ########################
    # Store folder
    string = re.split(r'/', name_piano)[-1]
    name_track = re.sub('piano_solo.mid', '', string)
    generated_folder = save_folder + '/fd_' + initialization_type + '_init/' + name_track
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)
    ########################

    ########################
    # Get trainer
    with open(os.path.join(config_folder_fd, 'which_trainer'), 'r') as ff:
        which_trainer_fd = ff.read()
    # Trainer
    trainer_fd = import_trainer(which_trainer_fd, model_parameters_fd, parameters)
    ########################

    ############################################################
    # Generate
    ############################################################
    time_generate_0 = time.time()
    model_path = 'model_accuracy'
    pr_orchestra_gen = generate(trainer_fd, pr_piano_gen_norm, silence_piano, duration_piano, config_folder_fd, model_path, pr_orchestra_seed, batch_size=number_of_version, mask_instrumentation=mask_instrumentation)
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

    ############################################################
    # Reconstruct and write
    ############################################################
    prefix_name = ''
    if initialization_type != 'seed':
        # remove padded zeros at the beginning
        pr_orchestra_gen = pr_orchestra_gen[:, seed_size:, :]
        pr_piano_gen = pr_piano_gen[seed_size:, :]
        if parameters['temporal_granularity'] == 'event_level':
            event_piano = event_piano[:-seed_size]
        seed_size = 0
    generation_utils.reconstruct_generation(pr_orchestra_gen, event_piano, generated_folder, prefix_name, parameters, seed_size)
    generation_utils.reconstruct_original(pr_piano_gen, pr_orchestra_truth, event_piano, generated_folder, parameters)
    return

if __name__ == '__main__':
    import LOP.Scripts.config as config

    # score_source = "/Users/crestel/Recherche/databases/Orchestration/LOP_database_mxml_clean/liszt_classical_archives/16"
    # config_folder_fd = "/Users/crestel/Recherche/lop/LOP/Experiments_pres/LSTM_reference/reference/0"
    # save_folder = "../../DEBUG/"

    model_path = "/Users/crestel/Recherche/lop/LOP/Experiments_pres/LSTM_reference/"
    save_folder_root = "../../DEBUG"
    config_folder_fds = [model_path + '/' + str(e) for e in range(10)]
    
    for config_ind, config_folder_fd in enumerate(config_folder_fds):
        score_sources = []
        with open(config_folder_fd + '/test_names.txt', 'r') as ff:
            for row in ff:
                score_sources.append(config.database_root() + "/" + row.strip('\n'))
        logger = logging.getLogger('worker')
        logger.setLevel(logging.INFO)

        save_folder = save_folder_root + '/' + str(config_ind)
        for initialization_type in ["zeros"]:
            for score_source in score_sources:
                generate_midi(config_folder_fd, score_source, save_folder, initialization_type=initialization_type, number_of_version=3, duration_gen=100, logger_generate=logger)


    # config_folder_fds = '/fast-1/leo/automatic_arrangement/Saved_models/LSTM_fd/0'
    
    # logger = logging.getLogger('worker')
    # logger.setLevel(logging.INFO)
    # generate_midi(config_folder_fd, score_source, save_folder, initialization_type="zeros", number_of_version=3, duration_gen=100, logger_generate=logger)