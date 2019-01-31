#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

_EPSILON = 1e-20

def accuracy_measure(true_frame, pred_frame):    
    axis = len(true_frame.shape) - 1

    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive

    accuracy_measure = np.where(quotient==0., 1., np.true_divide(true_positive, (quotient+_EPSILON)))

    return accuracy_measure


def accuracy_measure_test(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1

    C = 1.8
    K = 1.8 + 500*0.01 + 2*(1-0.8)
    # C = 0
    # K = 0
    
    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive
    
    accuracy_measure = np.where(quotient==0., 1., np.true_divide(true_positive + C, quotient + K))

    return accuracy_measure


def accuracy_measure_test_2(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1

    C = 1.8
    K = 1.8 + 500*0.01 + 2*(1-0.8)
    # C = 0
    # K = 0
    
    pred_frame_log = np.log(pred_frame + _EPSILON)
    non_pred_frame_log = np.log(1 - pred_frame + _EPSILON)
    
    true_positive = np.sum(pred_frame_log * true_frame, axis=axis)
    false_negative = np.sum(non_pred_frame_log * true_frame, axis=axis)
    false_positive = np.sum(pred_frame_log * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive
    
    accuracy_measure = np.where(quotient==0., 1., np.log(np.true_divide(true_positive + C, quotient + K)))

    return accuracy_measure


def true_accuracy_measure(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1

    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    true_negative = np.sum((1-pred_frame) * (1-true_frame), axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive + true_negative

    accuracy_measure = np.true_divide((true_negative + true_positive), (quotient + _EPSILON))

    return accuracy_measure


def f_measure(true_frame, pred_frame):
    precision = precision_measure(true_frame, pred_frame)
    recall = recall_measure(true_frame, pred_frame)
    f_measure = 2 * np.true_divide((precision * recall), (precision + recall + _EPSILON))
    return f_measure


def accuracy_measure_continuous(true_frame, pred_frame):
    threshold = 0.05
    diff_abs = np.absolute(true_frame - pred_frame)
    diff_thresh = np.where(diff_abs < threshold, 1, 0)
    binary_truth = np.where(pred_frame > 0., 1, 0)
    tp = np.sum(diff_thresh * binary_truth, axis=-1)
    false = np.sum((1 - diff_thresh), axis=-1)

    quotient = tp + false

    accuracy_measure =  np.where(quotient==0., 1, np.true_divide(tp, quotient + _EPSILON))

    return accuracy_measure


def recall_measure(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1
    # true_frame must be a binary vector
    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)

    quotient = true_positive + false_negative + 1e-20

    recall_measure = np.where(quotient==0., 1., np.true_divide(true_positive, (quotient + _EPSILON)))

    return recall_measure


def precision_measure(true_frame, pred_frame):
    axis = true_frame.ndim - 1
    # true_frame must be a binary vector
    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_positive + 1e-20

    precision_measure = np.where(quotient==0., 1., np.true_divide(true_positive, (quotient + _EPSILON)))

    return precision_measure

def binary_cross_entropy(true_frame, pred_frame):
    axis = true_frame.ndim - 1
    cross_entr_dot = np.multiply(true_frame, np.log(pred_frame + _EPSILON)) + 0.3 * np.multiply((1-true_frame), np.log((1-pred_frame+_EPSILON)))
    # Mean over feature dimension
    cross_entr = - np.mean(cross_entr_dot, axis=axis)
    return cross_entr

def compute_numerical_gradient(measure_fun, true_frame, pred_frame):
    num_batch, num_dim = pred_frame.shape
    # Cost function are one dimensional
    numeric_grad = np.zeros((num_batch, num_dim))
    for i in range(num_dim):
        ei = np.zeros((num_batch, num_dim))
        ei[:, i] = 1
        f_neg = measure_fun(true_frame, pred_frame - _EPSILON * ei)
        f_pos = measure_fun(true_frame, pred_frame + _EPSILON * ei)
        numeric_grad[:, i] = (f_pos - f_neg) / (2 * _EPSILON)
    return numeric_grad


def weighted_accuracy_test(true_frame, pred_frame, lamb=1):
    axis = len(true_frame.shape) - 1

    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    true_negative = np.sum((1-pred_frame) * (1-true_frame), axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    dividend = true_positive + lamb * true_negative
    divisor = true_positive + false_negative + false_positive + lamb*true_negative

    accuracy_measure = np.true_divide(dividend, divisor)

    return accuracy_measure


if __name__=='__main__':
    
    # Create a grid
    pred_frame = []
    for x in range(0.1, 0.9, 0.05):
        for y in range(0.1, 0.9, 0.05):
            pred_frame.append((x,y))
    pred_frame = np.asarray(pred_frame)
    
    true_frame_0_0 = np.zeros((1, 2))
    true_frame_0_1 = np.zeros((1, 2)); true_frame_0_0[0, 0] = 1
    true_frame_1_1 = np.ones((1, 2))
    
    numeric_gradient = compute_numerical_gradient(accuracy_measure_test, true_frame, pred_frame, 1e-4)