#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf


def embedding(inputs, emb_init, scope='Embedding', reuse=None):
    with tf.device('cpu'), tf.variable_scope(scope, reuse=reuse):
        embed = tf.Variable(emb_init, dtype=tf.float32, name='Embedding')
        return tf.nn.embedding_lookup(embed, inputs)


def gnn(inputs, adjacency, dim, layer, scope='GNN', reuse=None):
    """
    graphs nn on compounds structure
    :param inputs: [n, dim]
    :param adjacency: compound atom adjacency matrix [n, n], n is the number of atoms
    :param dim: embedding size 10
    :param layer: gnn layer num
    :param scope: tf variable scope name
    :param reuse: tf.AUTO_REUSE
    :return: [1, dim]
    """
    with tf.variable_scope(scope, reuse=reuse):
        gnn_out = inputs
        for i in range(layer):
            hidden = tf.layers.dense(gnn_out, dim, name='gnn_' + str(i), reuse=reuse)
            gnn_out += tf.matmul(adjacency, hidden)
        return tf.expand_dims(tf.reduce_mean(gnn_out, 0), 0)


def attention_cnn(inputs, attn, filter_size, dim, n_filters, layer, scope='AttnCNN', reuse=None):
    """
    cnn on protein sequences
    :param inputs: [n, dim] n is the number of protein sub sequences
    :param attn: [1, dim]
    :param filter_size: conv window size
    :param dim: embedding size 10
    :param n_filters: out channel 1
    :param layer: cnn layer num
    :param scope: tf variable scope name
    :param reuse: tf.AUTO_REUSE
    :return: [1, dim]
    """
    with tf.variable_scope(scope, reuse=reuse):
        cnn_out = tf.expand_dims(tf.expand_dims(inputs, -1), 0)
        for i in range(layer):
            cnn_out = tf.layers.conv2d(cnn_out, n_filters, [filter_size, dim], padding='same',
                                       activation=tf.nn.relu, name='cnn_' + str(i), reuse=reuse)
        cnn_out = tf.squeeze(cnn_out, -1)
        hidden_out = tf.layers.dense(cnn_out, dim, name='h_out')  # [1, ?, dim]
        hidden_out = tf.squeeze(hidden_out, 0)  # [?, dim]
        hidden_attn = tf.layers.dense(attn, dim, name='h_attn')  # [1, dim]
        weights = tf.nn.tanh(tf.matmul(hidden_out, tf.transpose(hidden_attn)))  # [?, 1]
        out = tf.matmul(tf.transpose(weights), hidden_out)  # [1, dim]
        return out


def linear(inputs, label_num, linear_size, scope='Linear', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        linear_out = inputs
        for size in linear_size:
            linear_out = tf.layers.dense(linear_out, size, activation=tf.nn.relu)
        return tf.layers.dense(linear_out, label_num, activation=None)




















