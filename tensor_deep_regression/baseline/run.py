#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
import os
from data_loader import FlyingChairDataset
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

BATCH_SIZE = 128
PREFETCH_SIZE = 1
NR_PROC = 4
IMAGE_SHAPE = (64, 64)
NR_CHANNEL = 3
OUT_DIM = 4

import tensorflow as tf
from tensorpack.train import TrainConfig, QueueInputTrainer
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.utils import *
from tensorpack.tfutils import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import *


class Model(ModelDesc):
    def _get_input_vars(self):
        return [
            InputVar(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NR_CHANNEL], 'left'),
            InputVar(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NR_CHANNEL], 'right'),
            InputVar(tf.float32, [None, OUT_DIM], 'label'),
        ]

    def _build_graph(self, input_vars):
        l_image, r_image, label = input_vars
        if get_current_tower_context().is_training:
            tf.summary.image('train_left_image', l_image, BATCH_SIZE)
            tf.summary.image('train_right_image', r_image, BATCH_SIZE)

        # build model
        nl = tf.tanh
        dropout_keep_prob = 0.5

        def vgg_pipeline(input_tensor, name='', nl=nl):
            with argscope(Conv2D, nl=nl, use_bias=False, padding='SAME'):
                l = Conv2D('conv0' + name, input_tensor, kernel_shape=3, out_channel=32)
                l = MaxPooling('pool0' + name, l, 2)
                l = Conv2D('conv1' + name, input_tensor, kernel_shape=3, out_channel=64)
                l = MaxPooling('pool1' + name, l, 2)
                l = Conv2D('conv2' + name, input_tensor, kernel_shape=3, out_channel=96)
                l = MaxPooling('pool2' + name, l, 2)
                l = Conv2D('conv3' + name, input_tensor, kernel_shape=3, out_channel=128)
                l = MaxPooling('pool3' + name, l, 2)
                return l
        l0 = vgg_pipeline(l_image, name='left')
        l1 = vgg_pipeline(r_image, name='right')
        l = tf.concat_v2([l0, l1], 3, name='concat')
        l = FullyConnected('fc0', l, out_dim=256, nl=nl)
        l = tf.nn.dropout(l, dropout_keep_prob, name='dropout_fc0')
        l = FullyConnected('fc1', l, out_dim=128, nl=tf.identity)
        l = FullyConnected('fct', l, out_dim=OUT_DIM, nl=tf.identity)
        l = tf.nn.sigmoid(l, name='pred')

        # loss
        cost = tf.reduce_mean(tf.square(l - label))
        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.004,
                         regularize_cost('.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)
        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        cost = tf.add_n([cost, wd_cost], name='cost')

        self.cost = cost


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = FlyingChairDataset(train_or_test)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, PREFETCH_SIZE, NR_PROC)
    return ds


def get_config():
    # prepare dataset
    step_per_epoch = 1024
    dataset_train = get_data('train')
    dataset_test = get_data('test')

    sess_config = get_default_sess_config(0.5)

    nr_gpu = get_nr_gpu()
    lr = tf.train.exponential_decay(
        learning_rate=1e-5,
        global_step=get_global_step_var(),
        decay_steps=step_per_epoch * 10,
        decay_rate=0.8,
        staircase=True, name='learning_rate')
    tf.summary.scalar('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            #InferenceRunner(dataset_test, ClassificationError())
        ]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=1000,
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    logger.set_logger_dir(os.path.join(BASE_DIR, 'train_log'))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        QueueInputTrainer(config).train()
