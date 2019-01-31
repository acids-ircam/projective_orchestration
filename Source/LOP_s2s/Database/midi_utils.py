#!/usr/bin/env python
# -*- coding: utf8 -*-

def pitch_to_pc(pitch):
    # Pitch to pitch-class
    pc = pitch % 12 
    octave = int(pitch / 12)    # Floor it
    return pc, octave

def pitch_to_pc(pitch):
    # Pitch to pitch-class
    pc = pitch % 12 
    octave = int(pitch / 12)    # Floor it
    return pc, octave
    
def pitch_octave_instru(elem):
    return elem[0], elem[1], elem[3]