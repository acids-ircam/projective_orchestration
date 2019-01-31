#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np


def up_criterion(val_tab, epoch, number_strips=3, validation_order=2):
    #######################################
    # Article
    # Early stopping, but when ?
    # Lutz Prechelt
    # UP criterion
    UP = True
    OVERFITTING = True
    s = 0
    sum_score = 0
    epsilon = 0.001
    while(UP and s < number_strips):
        t = epoch - s
        tmk = epoch - s - validation_order

        if not val_tab[t] == 0:  # Deal with zero score validation models
            UP = val_tab[t] > val_tab[tmk] - epsilon * abs(val_tab[tmk])   # Avoid extremely small evolutions

        s += validation_order
        sum_score += val_tab[t]
        # s += 1
        if not UP:
            OVERFITTING = False
    return OVERFITTING

def check_for_nan(val_tab, measures_to_check, max_nan):
    isnan = 0
    for measure in measures_to_check:
        isnan = max(isnan, np.sum(np.isnan(val_tab[measure])))
    return (isnan > 3)