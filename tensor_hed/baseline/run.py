#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
import os
import data_loader
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

BATCH_SIZE = 1
PREFETCH_SIZE = 4
NR_PROC = 1
IMAGE_SHAPE = (256, 256)
NR_CHANNEL = 3
OUT_CHANNEL = 1

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
            InputVar(tf.float32, [BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NR_CHANNEL], 'input'),
            InputVar(tf.float32, [BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], OUT_CHANNEL], 'label'),
        ]

    def _get_cost(self, input_vars, is_training):
        image, label = input_vars

        if is_training:
            tf.image_summary("train_image", image, BATCH_SIZE)

        with argscope(Conv2D, nl=BNReLU(is_training), use_bias=False, kernel_shape=3):
            l = Conv2D('out', image, out_channel=OUT_CHANNEL, padding='SAME')

        def cross_entropy(z, y):
            """
            :param z: output of nn
            :param y: ground truth
            """
            z = tf.reshape(z, tf.pack([tf.shape(z)[0], -1]))
            y = tf.reshape(y, tf.pack([tf.shape(y)[0], -1]))

            count_neg = tf.reduce_sum(1. - y)
            count_pos = tf.reduce_sum(y)
            total = tf.add(count_neg, count_pos)
            beta = tf.truediv(count_neg, total)

            eps = 1e-8
            loss_pos = tf.mul(-beta, tf.reduce_sum(tf.mul(tf.log(tf.abs(z) + eps), y), 1))
            loss_neg = tf.mul(1. - beta, tf.reduce_sum(tf.mul(tf.log(tf.abs(1. - z) + eps), 1. - y), 1))
            cost = tf.sub(loss_pos, loss_neg)
            cost = tf.reduce_mean(cost, name='cost')

            return cost

        cost = cross_entropy(l, label)
        #tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        return cost


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = data_loader.get_dataset(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.BrightnessAdd(15),
            imgaug.Contrast((0.8, 1.2)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, PREFETCH_SIZE, NR_PROC)
    return ds


def get_config():
    # prepare dataset
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    step_per_epoch = 1024

    sess_config = get_default_sess_config(0.5)

    nr_gpu = get_nr_gpu()
    lr = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=get_global_step_var(),
        decay_steps=step_per_epoch * (30 if nr_gpu == 1 else 20),
        decay_rate=0.5, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

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
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
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
