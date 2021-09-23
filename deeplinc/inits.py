#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import tensorflow as tf
import numpy as np


def zeros(shape, name=None):
    """
    All zeros padding
    """
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """
    All ones padding
    """
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot_bengio(shape, name=None):
    """
    Weight initialization method referred to Glorot & Bengio (AISTATS 2010)
    """
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def uniform(shape, scale=0.05, name=None):
    """
    Uniform initialization
    """
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

