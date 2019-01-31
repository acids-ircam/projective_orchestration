#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import numpy as np

import LOP.Database.build_data_aux as build_data_aux
import LOP_database.utils.event_level as event_level
import LOP_database.utils.reconstruct_pr as reconstruct_pr
import LOP.Utils.process_data as process_data
import LOP_database.utils.pianoroll_processing as pianoroll_processing 
import LOP_database.utils.time_warping as time_warping

from LOP_database.midi.read_midi import Read_midi
from LOP_database.midi.write_midi import write_midi



def load_from_pair(tracks_path, quantization, binarize_piano, binarize_orch, temporal_granularity, align_bool):
    ############################################################
    # Read piano midi file and orchestra score if defined
    ############################################################
    pr_piano, event_piano, duration_piano, name_piano, pr_orch, _, _, _, duration =\
        build_data_aux.process_folder(tracks_path, quantization, binarize_piano, binarize_orch, temporal_granularity, align_bool=align_bool)
    return pr_piano, event_piano, duration_piano, name_piano, pr_orch, duration

def load_solo(piano_midi, quantization, binarize_piano, temporal_granularity):
    # Read piano pr
    pr_piano = Read_midi(path, quantization).read_file()
    # Process pr_piano
    pr_piano = process_data.process_data_piano(pr_piano, binarize_piano)
    # Take event level representation
    if temporal_granularity == 'event_level':
        event_piano = event_level.get_event_ind_dict(pr_piano)
        pr_piano = time_warping.warp_pr_aux(pr_piano, event_piano)
    else:
        event_piano = None

    name_piano = re.sub(r'/.*\.mid', '', piano_midi)

    duration = pianoroll_processing.get_pianoroll_time(pr_piano)

    return pr_piano, event_piano, name_piano, None, None, duration

def init_with_seed(pr_orch, batch_size, seed_size, N_orchestra, instru_mapping):
    out_shape = (seed_size, N_orchestra)
    pr_orchestra_gen = np.zeros(out_shape, dtype=np.float32)
    orch_seed_beginning = {k: v[:seed_size] for k, v in pr_orch.items()}
    pr_orchestra_gen = build_data_aux.cast_small_pr_into_big_pr(orch_seed_beginning, 0, seed_size, instru_mapping, pr_orchestra_gen)
    # Stack number of batch
    pr_orchestra_gen_stacked = [pr_orchestra_gen for _ in range(batch_size)]
    pr_orchestra_gen_stacked = np.stack(pr_orchestra_gen_stacked, axis=0)
    return pr_orchestra_gen_stacked

def init_with_zeros(batch_size, seed_size, N_orchestra):
    out_shape = (batch_size, seed_size, N_orchestra)
    pr_orchestra_gen = np.zeros(out_shape, dtype=np.float32)
    return pr_orchestra_gen

def init_with_constant(batch_size, seed_size, N_orchestra, const_value):
    out_shape = (batch_size, seed_size, N_orchestra)
    pr_orchestra_gen = np.zeros(out_shape, dtype=np.float32) + const_value
    return pr_orchestra_gen

def init_with_random(batch_size, seed_size, N_orchestra, proba):
    out_shape = (batch_size, seed_size, N_orchestra)
    proba_np = np.zeros(out_shape, dtype=np.float32) + 0.5
    pr_orchestra_gen = np.float32(np.random.binomial(1, proba_np))
    return pr_orchestra_gen

def reconstruct_generation(generated_sequences, event_piano, generated_folder, prefix_name, parameters, seed_size):
    for write_counter in range(generated_sequences.shape[0]):
        # To distinguish when seed stop, insert a sustained note
        this_seq = generated_sequences[write_counter] * 127
        this_seq[:seed_size, 0] = 20
        # Reconstruct
        if parameters['temporal_granularity']=='event_level':
            pr_orchestra_rhythm = event_level.from_event_to_frame(this_seq, event_piano)
            pr_orchestra_rhythm_I = reconstruct_pr.instrument_reconstruction(pr_orchestra_rhythm, parameters["instru_mapping"])
            write_path = generated_folder + '/' + prefix_name + str(write_counter) + '_generated_rhythm.mid'
            write_midi(pr_orchestra_rhythm_I, parameters["quantization"], write_path, tempo=80)
        pr_orchestra_event = this_seq
        pr_orchestra_event_I = reconstruct_pr.instrument_reconstruction(pr_orchestra_event, parameters["instru_mapping"])
        write_path = generated_folder + '/' + prefix_name + str(write_counter) + '_generated.mid'
        if parameters['temporal_granularity']=='event_level':
            write_midi(pr_orchestra_event_I, 1, write_path, tempo=80)
        else:
            write_midi(pr_orchestra_event_I, parameters['quantization'], write_path, tempo=80)
    return

def reconstruct_original(pr_piano_gen, pr_orchestra_truth, event_piano, generated_folder, parameters):
    if parameters["temporal_granularity"]=='event_level':
        # Write original orchestration and piano scores, but reconstructed version, just to check
        A_rhythm = event_level.from_event_to_frame(pr_piano_gen, event_piano)
        B_rhythm = A_rhythm * 127
        piano_reconstructed_rhythm = reconstruct_pr.instrument_reconstruction_piano(B_rhythm, parameters["instru_mapping"])
        write_path = generated_folder + '/piano_reconstructed_rhythm.mid'
        write_midi(piano_reconstructed_rhythm, parameters["quantization"], write_path, tempo=80)
        # # Truth
        # A_rhythm = event_level.from_event_to_frame(pr_orchestra_truth, event_piano)
        # B_rhythm = A_rhythm * 127
        # orchestra_reconstructed_rhythm = reconstruct_pr.instrument_reconstruction(B_rhythm, parameters["instru_mapping"])
        # write_path = generated_folder + '/orchestra_reconstructed_rhythm.mid'
        # write_midi(orchestra_reconstructed_rhythm, parameters["quantization"], write_path, tempo=80)
        #
        A = pr_piano_gen
        B = A * 127
        piano_reconstructed = reconstruct_pr.instrument_reconstruction_piano(B, parameters["instru_mapping"])
        write_path = generated_folder + '/piano_reconstructed.mid'
        write_midi(piano_reconstructed, 1, write_path, tempo=80)
        #
        A = pr_orchestra_truth
        B = A * 127
        orchestra_reconstructed = reconstruct_pr.instrument_reconstruction(B, parameters["instru_mapping"])
        write_path = generated_folder + '/orchestra_reconstructed.mid'
        write_midi(orchestra_reconstructed, 1, write_path, tempo=80)
    else:
        A = pr_piano_gen
        B = A * 127
        piano_reconstructed = reconstruct_pr.instrument_reconstruction_piano(B, parameters["instru_mapping"])
        write_path = generated_folder + '/piano_reconstructed.mid'
        write_midi(piano_reconstructed, parameters["quantization"], write_path, tempo=80)
        #
        A = pr_orchestra_truth
        B = A * 127
        orchestra_reconstructed = reconstruct_pr.instrument_reconstruction(B, parameters["instru_mapping"])
        write_path = generated_folder + '/orchestra_reconstructed.mid'
        write_midi(orchestra_reconstructed, parameters["quantization"], write_path, tempo=80)
    return