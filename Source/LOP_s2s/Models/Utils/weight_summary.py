#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import re

def variable_summary(var, collections, plot_bool=False):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    name_var = re.split('/', var.name)[-1]
    name_var = re.split(':', name_var)[0]
    with tf.name_scope(name_var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections=collections)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections=collections)
        tf.summary.scalar('max', tf.reduce_max(var), collections=collections)
        tf.summary.scalar('min', tf.reduce_min(var), collections=collections)
        tf.summary.histogram('histogram', var, collections=collections)
        if plot_bool:
            # Reshape to fit the summay.image shape constraints
            if len(var.shape)==2:
                shape_var=var.get_shape()
                reshape_var = tf.reshape(var, [1, shape_var[0], shape_var[1], 1])
            else:
                import pdb; pdb.set_trace()
            tf.summary.image('im_weights', reshape_var, 1, collections=collections)
    return

def keras_layer_summary(layer, collections, plot_bool=False):
    for var in layer.trainable_weights:
        variable_summary(var, collections, plot_bool)
    return