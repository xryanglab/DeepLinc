#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


from .layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in list(kwargs.keys()):
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in list(kwargs.keys()):
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class Deeplinc(Model):
    """
    Referred to 
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, hidden1_dim, hidden2_dim, **kwargs):
        super(Deeplinc, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.h1_dim = hidden1_dim
        self.h2_dim = hidden2_dim        
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder'):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=self.h1_dim,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)

            self.h1 = self.hidden1

            self.z_mean = GraphConvolution(input_dim=self.h1_dim,
                                           output_dim=self.h2_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(self.hidden1)

            self.z_log_std = GraphConvolution(input_dim=self.h1_dim,
                                              output_dim=self.h2_dim,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging,
                                              name='e_dense_3')(self.hidden1)

            self.z = self.z_mean + tf.random_normal([self.n_samples, self.h2_dim]) * tf.exp(self.z_log_std)

            self.reconstructions = InnerProductDecoder(input_dim=self.h2_dim,
                                          act=lambda x: x,
                                          logging=self.logging)(self.z)
            self.embeddings = self.z


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


class Discriminator(Model):
    def __init__(self, inputdim, dc_hidden1_dim, dc_hidden2_dim, **kwargs):  #注意input_dim和Deeplinc函数的hidden2_dim要一样
        super(Discriminator, self).__init__(**kwargs)

        self.act = tf.nn.relu
        self.input_dim = inputdim
        self.dc_h1_dim = dc_hidden1_dim
        self.dc_h2_dim = dc_hidden2_dim

    def construct(self, inputs, reuse = False):
        # with tf.name_scope('Discriminator'):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, self.input_dim, self.dc_h1_dim, name='dc_den1'))  #125,150
            dc_den2 = tf.nn.relu(dense(dc_den1, self.dc_h1_dim, self.dc_h2_dim, name='dc_den2'))  #150,125
            output = dense(dc_den2, self.dc_h2_dim, 1, name='dc_output')
            return output
