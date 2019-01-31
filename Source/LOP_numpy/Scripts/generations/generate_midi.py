#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import logging
import pickle as pkl
import re
import numpy as np
import os
import time
try:
    import torch
except:
    pass

from LOP.Scripts.generations.generate import generate

import LOP.Database.build_data_aux as build_data_aux
from LOP.Utils.process_data import process_data_piano, process_data_orch

from LOP_database.midi.read_midi import Read_midi
from LOP_database.midi.write_midi import write_midi
from LOP_database.utils.pianoroll_processing import get_pianoroll_time, extract_pianoroll_part
from LOP_database.utils.event_level import get_event_ind_dict, from_event_to_frame
from LOP_database.utils.time_warping import warp_pr_aux
from LOP_database.utils.reconstruct_pr import instrument_reconstruction, instrument_reconstruction_piano
from LOP.Scripts.import_functions.import_trainer import import_trainer

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


def generate_midi(config_folder, score_source, number_of_version, duration_gen, logger_generate):
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
    ############################################################
    # Load model, config and data
    ############################################################

    ########################
    # Load config and model
    parameters = pkl.load(open(config_folder + '/script_parameters.pkl', 'rb'))
    model_parameters = pkl.load(open(config_folder + '/model_params.pkl', 'rb'))
    # Set a minimum seed size, because for very short models you don't event see the beginning
    seed_size = max(model_parameters['temporal_order'], 10) - 1
    quantization = parameters['quantization']
    temporal_granularity = parameters['temporal_granularity']
    instru_mapping = parameters['instru_mapping']
    ########################

    #######################
    # Load data
    if re.search(r'mid$', score_source):
        pr_piano, event_piano, duration_piano, name_piano, pr_orch, instru_orch, duration = load_solo(score_source, quantization, parameters["binarize_piano"], temporal_granularity)
    else:
        pr_piano, event_piano, duration_piano, name_piano, pr_orch, instru_orch, duration = load_from_pair(score_source, quantization, parameters["binarize_piano"], parameters["binarize_orch"], temporal_granularity)

    if (duration is None) or (duration<duration_gen):
        logger_generate.info("Track too short to be used")
        return
    ########################

    ########################
    # Shorten
    # Keep only the beginning of the pieces (let's say a 100 events)
    pr_piano = extract_pianoroll_part(pr_piano, 0, duration_gen)
    if parameters["duration_piano"]:
        duration_piano = np.asarray(duration_piano[:duration_gen])
    else:
        duration_piano = None
    if parameters["temporal_granularity"] == "event_level":
        event_piano = event_piano[:duration_gen]
    pr_orch = extract_pianoroll_part(pr_orch, 0, duration_gen)
    ########################

    ########################
    # Instanciate piano pianoroll
    N_piano = instru_mapping['Piano']['index_max']
    pr_piano_gen = np.zeros((duration_gen, N_piano), dtype=np.float32)
    pr_piano_gen = build_data_aux.cast_small_pr_into_big_pr(pr_piano, {}, 0, duration_gen, instru_mapping, pr_piano_gen)
    pr_piano_gen_flat = pr_piano_gen.sum(axis=1)
    silence_piano = [e for e in range(duration_gen) if pr_piano_gen_flat[e]== 0]
    ########################

    ########################
    # Instanciate orchestra pianoroll with orchestra seed
    N_orchestra = parameters['N_orchestra']
    if pr_orch:
        pr_orchestra_gen = np.zeros((seed_size, N_orchestra), dtype=np.float32)
        orch_seed_beginning = {k: v[:seed_size] for k, v in pr_orch.items()}
        pr_orchestra_gen = build_data_aux.cast_small_pr_into_big_pr(orch_seed_beginning, instru_orch, 0, seed_size, instru_mapping, pr_orchestra_gen)
        pr_orchestra_truth = np.zeros((duration_gen, N_orchestra), dtype=np.float32)
        pr_orchestra_truth = build_data_aux.cast_small_pr_into_big_pr(pr_orch, instru_orch, 0, duration_gen, instru_mapping, pr_orchestra_truth)
    else:
        pr_orchestra_gen = None
        pr_orchestra_truth = None
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
        piano_resize_emb[:, 0, instru_mapping['Piano']['pitch_min']:instru_mapping['Piano']['pitch_max']] = pr_piano_gen
        piano_resize_emb_TT = torch.tensor(piano_resize_emb)
        piano_embedded_TT = embedding_model(piano_resize_emb_TT.float(), 0)
        pr_piano_gen_embedded = piano_embedded_TT.numpy()
    else:
        pr_piano_gen_embedded = pr_piano_gen
    time_embedding = time.time() - time_embedding
    #######################################

    ########################
    # Inputs' normalization
    normalizer = pkl.load(open(os.path.join(config_folder, 'normalizer.pkl'), 'rb'))
    if parameters["embedded_piano"]:        # When using embedding, no normalization
        pr_piano_gen_norm = pr_piano_gen_embedded
    else:
        pr_piano_gen_norm = normalizer.transform(pr_piano_gen_embedded)
    ########################
    
    ########################
    # Store folder
    string = re.split(r'/', name_piano)[-1]
    name_track = re.sub('piano_solo.mid', '', string)
    generated_folder = config_folder + '/generation_reference_example/' + name_track
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)
    ########################

    ########################
    # Get trainer
    with open(os.path.join(config_folder, 'which_trainer'), 'r') as ff:
        which_trainer = ff.read()
    # Trainer
    trainer = import_trainer(which_trainer, model_parameters, parameters)
    
    ########################

    ############################################################
    # Generate
    ############################################################
    time_generate_0 = time.time()
    generated_sequences = {}
    for measure_name in parameters['save_measures']:
        model_path = 'model_' + measure_name
        generated_sequences[measure_name] = generate(trainer, pr_piano_gen_norm, silence_piano, duration_piano, config_folder, model_path, pr_orchestra_gen, batch_size=number_of_version)
        
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

    ############################################################
    # Reconstruct and write
    ############################################################
    def reconstruct_write_aux(generated_sequences, prefix):
        for write_counter in range(generated_sequences.shape[0]):
            # To distinguish when seed stop, insert a sustained note
            this_seq = generated_sequences[write_counter] * 127
            this_seq[:seed_size, 0] = 20
            # Reconstruct
            if parameters['temporal_granularity']=='event_level':
                pr_orchestra_rhythm = from_event_to_frame(this_seq, event_piano)
                pr_orchestra_rhythm_I = instrument_reconstruction(pr_orchestra_rhythm, instru_mapping)
                write_path = generated_folder + '/' + prefix + '_' + str(write_counter) + '_generated_rhythm.mid'
                write_midi(pr_orchestra_rhythm_I, quantization, write_path, tempo=80)
            pr_orchestra_event = this_seq
            pr_orchestra_event_I = instrument_reconstruction(pr_orchestra_event, instru_mapping)
            write_path = generated_folder + '/' + prefix + '_' + str(write_counter) + '_generated.mid'
            write_midi(pr_orchestra_event_I, 1, write_path, tempo=80)
        return

    for measure_name in parameters["save_measures"]:
        reconstruct_write_aux(generated_sequences[measure_name], measure_name)
    
    ############################################################
    ############################################################
    if parameters["temporal_granularity"]=='event_level':
        # Write original orchestration and piano scores, but reconstructed version, just to check
        A_rhythm = from_event_to_frame(pr_piano_gen, event_piano)
        B_rhythm = A_rhythm * 127
        piano_reconstructed_rhythm = instrument_reconstruction_piano(B_rhythm, instru_mapping)
        write_path = generated_folder + '/piano_reconstructed_rhythm.mid'
        write_midi(piano_reconstructed_rhythm, quantization, write_path, tempo=80)
        # Truth
        A_rhythm = from_event_to_frame(pr_orchestra_truth, event_piano)
        B_rhythm = A_rhythm * 127
        orchestra_reconstructed_rhythm = instrument_reconstruction(B_rhythm, instru_mapping)
        write_path = generated_folder + '/orchestra_reconstructed_rhythm.mid'
        write_midi(orchestra_reconstructed_rhythm, quantization, write_path, tempo=80)
        #
        A = pr_piano_gen
        B = A * 127
        piano_reconstructed = instrument_reconstruction_piano(B, instru_mapping)
        write_path = generated_folder + '/piano_reconstructed.mid'
        write_midi(piano_reconstructed, 1, write_path, tempo=80)
        #
        A = pr_orchestra_truth
        B = A * 127
        orchestra_reconstructed = instrument_reconstruction(B, instru_mapping)
        write_path = generated_folder + '/orchestra_reconstructed.mid'
        write_midi(orchestra_reconstructed, 1, write_path, tempo=80)
    else:
        A = pr_piano_gen
        B = A * 127
        piano_reconstructed = instrument_reconstruction_piano(B, instru_mapping)
        write_path = generated_folder + '/piano_reconstructed.mid'
        write_midi(piano_reconstructed, quantization, write_path, tempo=80)
        #
        A = pr_orchestra_truth
        B = A * 127
        orchestra_reconstructed = instrument_reconstruction(B, instru_mapping)
        write_path = generated_folder + '/orchestra_reconstructed.mid'
        write_midi(orchestra_reconstructed, quantization, write_path, tempo=80)
    ############################################################
    ############################################################
    return

if __name__ == '__main__':
    import config
    config_folder = '/fast-1/leo/automatic_arrangement/Results/Data_noC_bp_bo_tempGran100/LSTM_plugged_base/trAB_teA/0'
    score_sources = [
        # Bouliane valid
        config.database_root() + '/bouliane/16',
        # Bouliane train
        config.database_root() + '/bouliane/0',
        # Bouliane test
        config.database_root() + '/bouliane/17',
        # Spotify train
        config.database_root() + '/hand_picked_Spotify/0',
        # Spotify test
        config.database_root() + '/hand_picked_Spotify/21',
        # Spotify valid
        config.database_root() + '/hand_picked_Spotify/20',
        # Liszt train
        config.database_root() + '/liszt_classical_archives/0',
        # Liszt test
        config.database_root() + '/liszt_classical_archives/17',
        # Liszt valid
        config.database_root() + '/liszt_classical_archives/16'
    ]

    import logging
    logger = logging.getLogger('worker')
    logger.setLevel(logging.INFO)
    for score_source in score_sources:
        generate_midi(config_folder, score_source, number_of_version=3, duration_gen=100, logger_generate=logger)