#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from gnn.data_utils import get_now
from gnn.evaluation import get_auc, get_aupr
from gnn.modules import embedding, gnn, attention_cnn, linear


class Model(object):
    def __init__(self, model_path, lr, n_fingerprint, n_word, dim, layer_gnn,
                 layer_cnn, linear_size, drop_rate, filter_size):
        self.model_path = model_path

        with tf.device('cpu'):
            embed_fp = tf.Variable(tf.random_uniform([n_fingerprint, dim]))
            embed_word = tf.Variable(tf.random_uniform([n_word, dim]))

        self.data_smi = tf.placeholder(shape=[None], dtype=tf.int32)
        self.data_adj = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.data_seq = tf.placeholder(shape=[None], dtype=tf.int32)
        self.data_y = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        self.training = tf.placeholder(tf.bool)

        emb_smi = embedding(self.data_smi, embed_fp)
        emb_seq = embedding(self.data_seq, embed_word)

        smi_vec = gnn(emb_smi, self.data_adj, dim, layer_gnn, reuse=tf.AUTO_REUSE)
        seq_vec = attention_cnn(emb_seq, smi_vec, filter_size, dim, 1, layer_cnn, reuse=tf.AUTO_REUSE)

        feature = tf.concat([smi_vec, seq_vec], 1)
        out = tf.layers.dropout(feature, drop_rate, training=self.training)
        self.output = linear(out, 1, linear_size)

        self.loss = tf.losses.sigmoid_cross_entropy(self.data_y, self.output)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        self.saver = tf.train.Saver()

    def train_step(self, sess: tf.InteractiveSession, train_x: np.ndarray, train_y: np.ndarray):
        batch_loss, batch_num = 0, len(train_x)
        for i in range(batch_num):
            batch_x, batch_y = train_x[i], train_y[i]
            feed_dict = {
                self.data_smi: batch_x[0],
                self.data_adj: batch_x[1],
                self.data_seq: batch_x[2],
                self.data_y: np.array(batch_y).reshape([-1, 1]),
                self.training: True
            }
            _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            batch_loss += loss / batch_num
        return batch_loss

    def valid_step(self, sess: tf.InteractiveSession, valid_x: np.ndarray, valid_y: np.ndarray):
        batch_loss, batch_num, res = 0, len(valid_x), []
        for i in range(batch_num):
            batch_x, batch_y = valid_x[i], valid_y[i]
            feed_dict = {
                self.data_smi: batch_x[0],
                self.data_adj: batch_x[1],
                self.data_seq: batch_x[2],
                self.data_y: np.array(batch_y).reshape([-1, 1]),
                self.training: False
            }
            loss, output = sess.run([self.loss, self.output], feed_dict=feed_dict)
            batch_loss += loss / batch_num
            res.append(output[0])
        return batch_loss, res

    def predict_step(self, sess: tf.InteractiveSession, data_x: np.ndarray):
        batch_num, res = len(data_x), []
        for i in range(batch_num):
            feed_dict = {
                self.data_smi: data_x[i][0],
                self.data_adj: data_x[i][1],
                self.data_seq: data_x[i][2],
                self.training: False
            }
            output = sess.run(self.output, feed_dict=feed_dict)
            res.append(output[0])
        return res

    def train(self, sess: tf.InteractiveSession, train_x: np.ndarray, train_y: np.ndarray,
              valid_x: np.ndarray, valid_y: np.ndarray, epoch=20,
              batch_size=128, valid_batch_size=50, step=1, verbose=True):
        print(get_now(), 'start training')
        train_idx = sorted(range(len(train_x)), key=lambda x: len(train_x[x]), reverse=True)
        valid_idx = sorted(range(len(valid_x)), key=lambda x: len(valid_x[x]), reverse=True)
        sess.run(tf.global_variables_initializer())
        best_aupr, current = 0, 0
        for idx_epoch in range(epoch):
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i:i + batch_size]
                train_loss = self.train_step(sess, train_x[batch_idx], train_y[batch_idx])
                current += 1
                if current % step == 0:
                    print(get_now())
                    valid_loss, valid_res = 0, np.empty([len(valid_idx), 1], dtype=int)
                    for j in range(0, len(valid_idx), valid_batch_size):
                        valid_batch_idx = valid_idx[j:j + valid_batch_size]
                        loss, output = self.valid_step(sess, valid_x[valid_batch_idx], valid_y[valid_batch_idx])
                        valid_res[valid_batch_idx] = output
                        valid_loss += loss * len(valid_batch_idx)
                    valid_loss /= len(valid_idx)
                    auc, aupr = get_auc(valid_y, valid_res), get_aupr(valid_y, valid_res)
                    if aupr > best_aupr:
                        best_aupr = aupr
                        self.saver.save(sess, self.model_path)
                    if verbose:
                        print(get_now(), current, current * batch_size, idx_epoch, i + batch_size,
                              'train loss:', round(train_loss, 5),
                              'valid loss:', round(valid_loss, 5),
                              'AUC:', round(auc, 5), 'AUPR:', round(aupr, 5))

    def predict(self, sess: tf.InteractiveSession, data_x: np.ndarray, batch_size=128):
        print(get_now(), 'start predicting')
        self.saver.restore(sess, self.model_path)
        res = np.empty([len(data_x), 1])
        for i in range(0, len(data_x), batch_size):
            res[i:i + batch_size] = self.predict_step(sess, data_x[i:i + batch_size])
        return res
