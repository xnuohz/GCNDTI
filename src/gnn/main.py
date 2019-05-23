#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import click
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from ruamel.yaml import YAML
from gnn.model import Model
from gnn.data_utils import get_now, load_pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


@click.command()
@click.option('--data-cnf', help='dataset config')
@click.option('--model-cnf', help='model config')
def main(data_cnf, model_cnf):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_path = os.path.join(model_cnf['path'], model_cnf['name'] + '-' + data_cnf['name'])
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    model_name = model_cnf['name']
    print('model name:', model_name)

    fingerprint_dict = load_pickle(data_cnf['fingerprint_dict'])
    word_dict = load_pickle(data_cnf['word_dict'])

    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    print(get_now(), 'loading dataset')
    train_x, train_y = np.load(data_cnf['train']['input']), np.load(data_cnf['train']['label'])
    valid_x, valid_y = np.load(data_cnf['valid']['input']), np.load(data_cnf['valid']['label'])

    print('size of train set:', len(train_x))
    print('size of valid set:', len(valid_x))

    model = Model(**model_cnf['model'], model_path=model_path, n_fingerprint=n_fingerprint, n_word=n_word)

    model.train(sess, train_x, train_y, valid_x, valid_y, **model_cnf['train'])


if __name__ == '__main__':
    main()
