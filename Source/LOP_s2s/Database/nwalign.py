#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import numpy as np
import math


def score_function(ci, cj):
    # ci and cj are ensembles
    # set(
    #     (pc(12), octave(11), intensity(128), instru_integer, repeat_flag(1))
    #     (pc(12), octave(11), intensity(128), instru_integer, repeat_flag(1))
    #     )

    # Filter out information other than pc
    def filter_pc(cc):
        return set([e[0] for e in cc[1]])
    A = filter_pc(ci)
    B = filter_pc(cj)
    denominator = len(A|B)
    if denominator == 0:
        # Means both are silence
        return 1
    AandB = A & B
    AorB = A | B
    posTerm = len(AandB)
    negTerm = len(AorB - AandB)
    score = (posTerm - negTerm) / denominator

    # DIVISER PAR UNE CONSTANTE !!

    return score
    

def nwalign(seqj, seqi, gapOpen=-3, gapExtend=-1):
    """
    >>> global_align('COELANCANTH', 'PELICAN')
    ('COELANCANTH', '-PEL-ICAN--')

    nwalign must be used on list of pitch_class ensemble
    Two versions of the sequences are returned 
    First tuple is with removed elements (useful for training and initialize generation)
    Second tuple simply indicates where elemnts have been skipped in each sequence (useful for generating after a sequence has been init)

    TODO:
    Limit search zone
    Abort exploration if score becomes too small
    """

    UP, LEFT, DIAG, NONE = range(4)

    max_j = len(seqj)
    max_i = len(seqi)
    
    score   = np.zeros((max_i + 1, max_j + 1), dtype='f') - math.inf
    pointer = np.zeros((max_i + 1, max_j + 1), dtype='i')
    max_i, max_j

    pointer[0, 0] = NONE
    score[0, 0] = 0.0

    pointer[0, 1:] = LEFT
    pointer[1:, 0] = UP

    # Do we do that ?? Not sure...
    score[0, 1:] = gapExtend * np.arange(max_j)
    score[1:, 0] = gapExtend * np.arange(max_i)

    for i in range(1, max_i + 1):
        ci = seqi[i - 1]

        # Ok plus rapide, mais tester reconstruction
        j_min = max(i-1000,1)
        j_max = min(i+1000,max_j+1)
        for j in range(j_min, j_max):
        # for j in range(1, max_j + 1):
            cj = seqj[j - 1]
            
            termScore = score_function(ci, cj)
            diag_score = score[i - 1, j - 1] + termScore

            if pointer[i-1, j] == UP:
                up_score = score[i - 1, j] + gapExtend
            else:
                up_score = score[i - 1, j] + gapOpen
            
            if pointer[i,j-1] == LEFT:
                left_score = score[i, j - 1] + gapExtend
            else:
                left_score = score[i, j - 1] + gapOpen
            
            if diag_score >= up_score:
                if diag_score >= left_score:
                    score[i, j] = diag_score
                    pointer[i, j] = DIAG
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT

            else:
                if up_score > left_score:
                    score[i, j ]  = up_score
                    pointer[i, j] = UP
                else:
                    score[i, j]   = left_score
                    pointer[i, j] = LEFT
                    
                
    align_j = []
    align_i = []
    skip_j = []
    skip_i = []
    while True:
        p = pointer[i, j]
        if p == NONE: break
        s = score[i, j]
        if p == DIAG:
            align_j.append(seqj[j - 1])
            align_i.append(seqi[i - 1])
            i -= 1
            j -= 1
            skip_j.append(1)
            skip_i.append(1)
        elif p == LEFT:
            # Loose the element
            # align_j += seqj[j - 1]
            # align_i += [] #silence
            j -= 1
            skip_j.append(0)
        elif p == UP:
            # align_j += []
            # align_i += seqi[i - 1]
            i -= 1
            skip_i.append(0)
        else:
            raise Exception('wtf!')

    return (align_j[::-1], align_i[::-1]), (skip_j[::-1], skip_i[::-1])

if __name__ == "__main__":
    print(nwalign('COELANCANTH', 'PELICAN'))