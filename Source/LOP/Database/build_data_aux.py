#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import glob
import re
import csv
from unidecode import unidecode
import numpy as np
from LOP_database.midi.read_midi import Read_midi
from LOP_database.utils.pianoroll_processing import clip_pr, get_pianoroll_time
from LOP_database.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace, warp_pr_aux
from LOP_database.utils.event_level import get_event_ind_dict, from_event_to_frame
from LOP_database.utils.pianoroll_processing import sum_along_instru_dim
from LOP_database.utils.align_pianorolls import align_pianorolls
from LOP.Database.simplify_instrumentation import get_simplify_mapping
from LOP.Utils.process_data import process_data_piano, process_data_orch

from Musicxml_parser.scoreToPianoroll import mxml_to_pr

# DEBUG LIBS
from LOP_database.midi.write_midi import write_midi
import matplotlib.pyplot as plt

def get_instru_and_pr_from_folder_path(folder_path, quantization, clip=True):
    # There should be 2 files
    score_files = glob.glob(folder_path + '/*.xml')

    mapping_instru_mxml = get_simplify_mapping()
 
    # Time
    if len(score_files) != 2:
        raise Exception('There should be two xml files in ' + folder_path)

    def file_processing(path, quantization, clip):
        pianoroll, articulation, staccato_curve, _ = mxml_to_pr(path, quantization, mapping_instru_mxml, apply_staccato=True)
        # Clip
        if clip:
            pianoroll, articulation, staccato_curve = clip_pr(pianoroll, articulation, staccato_curve)
        return pianoroll, articulation, staccato_curve, get_pianoroll_time(pianoroll)

    pianoroll_0, articulation_0, staccato_curve_0, T0 = file_processing(score_files[0], quantization, clip)
    pianoroll_1, articulation_1, staccato_curve_1, T1 = file_processing(score_files[1], quantization, clip)

    # Files name, no extensions
    file_name_0 = re.sub(r'\.xml$', '', score_files[0])
    file_name_1 = re.sub(r'\.xml$', '', score_files[1])

    return pianoroll_0, articulation_0, staccato_curve_0, T0, file_name_0, pianoroll_1, articulation_1, staccato_curve_1, T1, file_name_1

def unmixed_instru(instru_string):
    instru_list = re.split(r' and ', instru_string)
    return instru_list

def instru_pitch_range(pr, instru_mapping, index_instru):
    for key, pr_instru in pr.items():
        # Get unmixed instru names
        instru_names = unmixed_instru(key)
        # Avoid mixed instrumentation for determining the range.
        # Why ?
        # For instance, tutti strings in the score will be written as a huge chord spanning all violin -> dbass range
        # Hence we don't want range of dbas in violin
        if len(instru_names) > 1:
            continue
        # Corresponding pianoroll
        if pr_instru.sum() == 0:
            continue
        for instru_name in instru_names:
            # Avoid "Remove" instruments
            if instru_name in instru_mapping.keys():
                old_max = instru_mapping[instru_name]['pitch_max']
                old_min = instru_mapping[instru_name]['pitch_min']
                # Get the min :
                #   - sum along time dimension
                #   - get the first non-zero index
                this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
                this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0]) + 1
                instru_mapping[instru_name]['pitch_min'] = min(old_min, this_min)
                instru_mapping[instru_name]['pitch_max'] = max(old_max, this_max)
            else:
                instru_mapping[instru_name] = {}
                this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
                this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0]) + 1
                instru_mapping[instru_name]['pitch_min'] = this_min
                instru_mapping[instru_name]['pitch_max'] = this_max
                if instru_name != 'Piano':
                    instru_mapping[instru_name]['index_instru'] = index_instru 
                    index_instru += 1

    return instru_mapping, index_instru


def clean_event(event, trace, trace_prod):
    # Remove from the traces the removed indices
    new_event = []
    counter = 0
    for t, tp in zip(trace, trace_prod):
        if t + tp == 2:
            new_event.append(event[counter])
            counter += 1
        elif t != 0:
            # the t=counter-th event is lost
            counter +=1
    return new_event

def discriminate_between_piano_and_orchestra(data_0, data_1):
    pr_0, _, _, _, _ = data_0
    pr_1, _, _, _, _ = data_1
    if len(set(pr_0.keys())) > len(set(pr_1.keys())):
        return data_1, data_0
    elif len(set(pr_0.keys())) < len(set(pr_1.keys())):
        return data_0, data_1
    else:
        # Both tracks have the same number of instruments
        return ([None] * 5, [None] * 5)

def cast_small_pr_into_big_pr(pr_small, time, duration, instru_mapping, pr_big):
    # Detremine x_min and x_max thanks to time and duration
    # Parse pr_small by keys (instrument)
    # Get insrument name in instru
    # For pr_instrument, remove the column out of pitch_min and pitch_max
    # Determine thanks to instru_mapping the y_min and y_max in pr_big

    # Detremine t_min and t_max thanks to time and duration
    t_min = time
    t_max = time + duration
    # Parse pr_small
    for track_name, pr_instru in pr_small.items():
        # Unmix instru (can be group of instrus separated by "and")
        track_name_processed = (track_name.rstrip('\x00')).replace('\r', '')
        instru_names = unmixed_instru(track_name_processed)
        for instru_name in instru_names:
            # "Remove" tracks
            # For pr_instrument, remove the column out of pitch_min and pitch_max
            try:
                pitch_min = instru_mapping[instru_name]['pitch_min']
                pitch_max = instru_mapping[instru_name]['pitch_max']
            except KeyError:
                print(instru_name + " instrument was not present in the training database")
                continue

            # Determine thanks to instru_mapping the y_min and y_max in pr_big
            index_min = instru_mapping[instru_name]['index_min']
            index_max = instru_mapping[instru_name]['index_max']

            # Insert the small pr in the big one :)
            # Insertion is max between already written notes and new ones
            try:
                pr_big[t_min:t_max, index_min:index_max] = np.maximum(pr_big[t_min:t_max, index_min:index_max], pr_instru[:, pitch_min:pitch_max])
            except:
                import pdb; pdb.set_trace()

    return pr_big

def process_folder(folder_path, quantization, binary_piano, binary_orch, temporal_granularity, gapopen=3, gapextend=1, align_bool=True):
    ##############################
    # Get instrus and prs from a folder name name
    pr0, articulation_0, staccato_0, T0, name0, pr1, articulation_1, staccato_1, T1, name1 = get_instru_and_pr_from_folder_path(folder_path, quantization)
    data_0 = (pr0, articulation_0, staccato_0, T0, name0)
    data_1 = (pr1, articulation_1, staccato_1, T1, name1)

    (pr_piano_X, articulation_piano, staccato_piano, T_piano, name_piano), \
    (pr_orch, articulation_orch, staccato_orch, T_orch, name_orch)=\
            discriminate_between_piano_and_orchestra(data_0, data_1)

    # if pr_contrabass[:, 62:].sum() > 1:
    #     import pdb; pdb.set_trace()

    # If corrupted files, pr_piano (and pr_orch) will be None
    if pr_piano_X is None:
        return [None] * 9

    # Remove from orch
    if "Remove" in pr_orch.keys():
        pr_orch.pop("Remove")
    # Group in piano
    pr_piano = {'Piano': sum_along_instru_dim(pr_piano_X)}

    # try:
    #     write_midi(pr_piano, ticks_per_beat=quantization, write_path="../DEBUG/test_piano.mid", articulation=articulation_piano)
    #     write_midi(pr_orch, ticks_per_beat=quantization, write_path="../DEBUG/test_orch.mid", articulation=articulation_orch)
        # write_midi({k: v*90 for k,v in pr_piano.items()}, ticks_per_beat=quantization, write_path="../DEBUG/test_piano.mid", articulation=articulation_piano)
        # write_midi({k: v*90 for k,v in pr_orch.items() if (v.sum()>0)}, ticks_per_beat=quantization, write_path="../DEBUG/test_orch.mid", articulation=articulation_orch)
    # except:
    #     print("Because of mixed instru cannot write reference")

    ##############################
    # Process pr (mostly binarized)
    pr_piano = process_data_piano(pr_piano, binary_piano)
    pr_orch = process_data_orch(pr_orch, binary_orch)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        event_piano = get_event_ind_dict(articulation_piano, pr_piano)
        event_orch = get_event_ind_dict(articulation_orch, pr_orch)
        def get_duration(event, last_time):
            start_ind = event[:]
            end_ind = np.zeros(event.shape, dtype=np.int)
            end_ind[:-1] = event[1:]
            end_ind[-1] = last_time
            duration_list = end_ind - start_ind
            return duration_list
        duration_piano = get_duration(event_piano, T_piano)
        duration_orch = get_duration(event_orch, T_orch)
        # Get the duration of each event
        pr_piano_event = warp_pr_aux(pr_piano, event_piano)
        pr_orch_event = warp_pr_aux(pr_orch, event_orch)
    else:
        event_piano = None
        event_orch = None
        duration_piano = None
        duration_orch = None
        pr_piano_event = pr_piano
        pr_orch_event = pr_orch

    ##############################
    ##############################
    # # Test for event-leve -> beat reconstruction
    # ##############################
    # # Instru mapping
    # import pickle as pkl
    # import LOP_database.utils.event_level as event_level
    # import LOP_database.utils.reconstruct_pr as reconstruct_pr
    # temp = pkl.load(open("/Users/crestel/Recherche/lop/LOP/Data/Data_A_ref_bp_bo_noEmb_tempGran32/temp.pkl", 'rb'))
    # instru_mapping = temp['instru_mapping']
    # N_orchestra = temp['N_orchestra']
    # N_piano = temp['instru_mapping']['Piano']['index_max']
    # matrix_orch = cast_small_pr_into_big_pr(pr_orch_event, 0, len(event_orch), instru_mapping, np.zeros((len(event_orch), N_orchestra)))
    # matrix_piano = cast_small_pr_into_big_pr(pr_piano_event, 0, len(event_piano), instru_mapping, np.zeros((len(event_piano), N_piano)))
    # ##############################
    # # Reconstruct rhythm
    # pr_orchestra_rhythm = event_level.from_event_to_frame(matrix_orch, event_orch)
    # pr_orchestra_rhythm_I = reconstruct_pr.instrument_reconstruction(pr_orchestra_rhythm, instru_mapping)
    # pr_piano_rhythm = event_level.from_event_to_frame(matrix_piano, event_piano)
    # pr_piano_rhythm_I = reconstruct_pr.instrument_reconstruction_piano(pr_piano_rhythm, instru_mapping)
    # ##############################
    # # Write midi
    # write_midi({k: v*90 for k,v in pr_piano_rhythm_I.items()}, ticks_per_beat=quantization, write_path="../DEBUG/test_piano_event.mid", articulation=articulation_piano)
    # write_midi({k: v*90 for k,v in pr_orchestra_rhythm_I.items() if (v.sum()>0)}, ticks_per_beat=quantization, write_path="../DEBUG/test_orch_event.mid", articulation=articulation_orch)
    ############################## 

    ##############################
    # Align tracks
    if align_bool:
        # piano_aligned, trace_piano, orch_aligned, trace_orch, trace_prod, total_time = align_pianorolls(pr_piano_event, pr_orch_event, gapopen, gapextend)
        piano_aligned, trace_piano, orch_aligned, trace_orch, trace_prod, total_time = align_pianorolls(pr_piano_event, pr_orch_event, gapopen, gapextend)
        # Clean events
        if (temporal_granularity == 'event_level'):
            if (trace_piano is None) or (trace_orch is None):
                event_piano_aligned = None
                event_orch_aligned = None
                duration_piano_aligned = None
                duration_orch_aligned = None
            else:
                event_piano_aligned = clean_event(event_piano, trace_piano, trace_prod)
                event_orch_aligned = clean_event(event_orch, trace_orch, trace_prod)
                duration_piano_aligned = clean_event(duration_piano, trace_piano, trace_prod)
                duration_orch_aligned = clean_event(duration_orch, trace_orch, trace_prod)
        else:
            event_piano_aligned = []
            event_orch_aligned = []
            duration_piano_aligned = []
            duration_orch_aligned = []
    else:
        piano_aligned = pr_piano_event
        event_piano_aligned = event_piano
        duration_piano_aligned = duration_piano
        orch_aligned = pr_orch_event
        event_orch_aligned = event_orch
        duration_orch_aligned = duration_orch
        total_time = T_piano
    ##############################

    ##############################
    ##############################
    # Test for aligned event Piano/Orch
    ##############################
    # Instru mapping
    # import pickle as pkl
    # import LOP_database.utils.event_level as event_level
    # import LOP_database.utils.reconstruct_pr as reconstruct_pr
    # temp = pkl.load(open("/Users/leo/Recherche/lop/LOP/Data/Data_DEBUG_bp_bo_noEmb_tempGran32/temp.pkl", 'rb'))
    # instru_mapping = temp['instru_mapping']
    # N_orchestra = temp['N_orchestra']
    # N_piano = temp['instru_mapping']['Piano']['index_max']
    # matrix_orch = cast_small_pr_into_big_pr(orch_aligned, 0, len(event_orch_aligned), instru_mapping, np.zeros((len(event_orch_aligned), N_orchestra)))
    # matrix_piano = cast_small_pr_into_big_pr(piano_aligned, 0, len(event_piano_aligned), instru_mapping, np.zeros((len(event_piano_aligned), N_piano)))
    # ##############################
    # # Reconstruct rhythm
    # pr_orchestra_I = reconstruct_pr.instrument_reconstruction(matrix_orch, instru_mapping)
    # pr_orchestra_rhythm = event_level.from_event_to_frame(matrix_orch, event_orch_aligned)
    # pr_orchestra_rhythm_I = reconstruct_pr.instrument_reconstruction(pr_orchestra_rhythm, instru_mapping)
    # #
    # pr_piano_I = reconstruct_pr.instrument_reconstruction_piano(matrix_piano, instru_mapping)
    # pr_piano_rhythm = event_level.from_event_to_frame(matrix_piano, event_piano_aligned)
    # pr_piano_rhythm_I = reconstruct_pr.instrument_reconstruction_piano(pr_piano_rhythm, instru_mapping)
    # ##############################
    # # Write midi
    # write_midi({k: v*90 for k,v in pr_piano_I.items()}, ticks_per_beat=1, write_path="../DEBUG/test_piano_event_aligned.mid", articulation=None)
    # write_midi({k: v*90 for k,v in pr_piano_rhythm_I.items()}, ticks_per_beat=quantization, write_path="../DEBUG/test_piano_rhythm_aligned.mid", articulation=None)
    # #
    # write_midi({k: v*90 for k,v in pr_orchestra_I.items() if (v.sum()>0)}, ticks_per_beat=1, write_path="../DEBUG/test_orch_event_aligned.mid", articulation=None)
    # write_midi({k: v*90 for k,v in pr_orchestra_rhythm_I.items() if (v.sum()>0)}, ticks_per_beat=quantization, write_path="../DEBUG/test_orch_rhythm_aligned.mid", articulation=None)
    # import pdb; pdb.set_trace()
    ##############################

    return piano_aligned, event_piano_aligned, duration_piano_aligned, name_piano, orch_aligned, event_orch_aligned, duration_orch_aligned, name_orch, total_time

if __name__ == '__main__':
    # path_to_folder = "/Users/leo/Recherche/databases/Orchestration/LOP_database_06_09_17/bouliane/0"
    path_to_folder = "/Users/leo/Recherche/databases/Orchestration/LOP_database_mxml/liszt_classical_archives/0"
    pr_piano, event_piano, duration_piano, name_piano, pr_orch, event_orch, duration_orch, name_orch, total_time = \
        process_folder(path_to_folder, 32, True, True, 'event_level')