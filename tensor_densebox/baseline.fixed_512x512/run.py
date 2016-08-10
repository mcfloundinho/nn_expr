#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
import os
import functools
from data_loader import WiderFaceDenseBoxDataset
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

"""
    python2 -m tensorpack.utils.loadcaffe PATH/TO/models/VGG/{VGG_ILSVRC_16_layers_deploy.prototxt,VGG_ILSVRC_16_layers.caffemodel} vgg16.npy
"""
VGG_PATH = os.path.join(BASE_DIR, '../vgg16.npy')

BATCH_SIZE = 1
PREFETCH_SIZE = 1
NR_PROC = 1
IMAGE_SHAPE = (512, 512)
NR_CHANNEL = 3
OUT_DIM = 5

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
    def IoULoss(self, pd, gt):
        mask = tf.cast(
            tf.greater(tf.reduce_sum(
                tf.cast(tf.greater(gt, 0), tf.int8), 3), 3),
            tf.float32
        )
        npd = tf.transpose(pd, [3, 0, 1, 2])
        ngt = tf.transpose(gt, [3, 0, 1, 2])
        area_x = tf.mul(
            tf.add(tf.gather(npd, 0), tf.gather(npd, 2)),
            tf.add(tf.gather(npd, 1), tf.gather(npd, 3)),
        )
        area_g = tf.mul(
            tf.add(tf.gather(ngt, 0), tf.gather(ngt, 2)),
            tf.add(tf.gather(ngt, 1), tf.gather(ngt, 3)),
        )
        w_overlap = tf.maximum(tf.constant(0, tf.float32), tf.add(
            tf.minimum(tf.gather(npd, 0), tf.gather(ngt, 0)),
            tf.minimum(tf.gather(npd, 2), tf.gather(ngt, 2)),
        ))
        h_overlap = tf.maximum(tf.constant(0, tf.float32), tf.add(
            tf.minimum(tf.gather(npd, 1), tf.gather(ngt, 1)),
            tf.minimum(tf.gather(npd, 3), tf.gather(ngt, 3)),
        ))
        area_overlap = tf.mul(w_overlap, h_overlap)
        area_u = tf.sub(tf.add(area_x, area_g), area_overlap)
        iou = tf.div(area_overlap, tf.add(area_u, tf.constant(1, tf.float32)))
        iou = tf.maximum(iou, tf.constant(1e-4, tf.float32))
        cost = -tf.log(iou)
        cost = tf.mul(cost, mask)
        cost = tf.reduce_sum(cost)
        return cost

    def _get_input_vars(self):
        return [
            #InputVar(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NR_CHANNEL], 'input'),
            #InputVar(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], OUT_DIM], 'label'),
            InputVar(tf.float32, [None, None, None, NR_CHANNEL], 'input'),
            InputVar(tf.float32, [None, None, None, OUT_DIM], 'label'),
        ]

    def _get_cost(self, input_vars, is_training):
        image, label = input_vars
        if is_training:
            tf.image_summary('train_image', image, BATCH_SIZE)

        # build model
        branch_list = []
        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, use_bias=False, padding='SAME'):
            with argscope(MaxPooling, stride=2, padding='VALID'):
                l = Conv2D('conv1_1', image, 64)
                l = Conv2D('conv1_2', l, 64)
                l = MaxPooling('pool1', l, 2)

                l = Conv2D('conv2_1', l, 128)
                l = Conv2D('conv2_2', l, 128)
                l = MaxPooling('pool2', l, 2)

                l = Conv2D('conv3_1', l, 256)
                l = Conv2D('conv3_2', l, 256)
                l = Conv2D('conv3_3', l, 256)
                l = MaxPooling('pool3', l, 2)

                l = Conv2D('conv4_1', l, 512)
                l = Conv2D('conv4_2', l, 512)
                l = Conv2D('conv4_3', l, 512)
                l = MaxPooling('pool4', l, 2)
                branch_list.append(l)

                l = Conv2D('conv5_1', l, 512)
                l = Conv2D('conv5_2', l, 512)
                l = Conv2D('conv5_3', l, 512)
                branch_list.append(l)

        outputs = []

        # confidence branch
        l = branch_list[0]
        l = Conv2D('dsn1-conv', l, out_channel=3, kernel_shape=3, padding='SAME', nl=tf.identity)
        for i in range(4):
            l = FixedUnPooling('dsn1-unpool{}'.format(i), l, 2)
            l = Conv2D('dsn1-unconv{}'.format(i), l, kernel_shape=3, out_channel=3, padding='SAME', nl=tf.identity)
        l = Conv2D('dsn1-conv-final', l, kernel_shape=3, out_channel=1, padding='SAME', nl=tf.identity)
        l = tf.nn.sigmoid(l, name='score')
        outputs.append(l)
        score_label = tf.slice(label, [0, 0, 0, 0], [-1, -1, -1, 1])
        score_cost = tf.reduce_mean(tf.square(l - score_label))

        # boxes branch
        l = branch_list[1]
        l = Conv2D('dsn2-conv', l, out_channel=OUT_DIM - 1, kernel_shape=3, padding='SAME', nl=tf.identity)
        for i in range(4):
            l = FixedUnPooling('dsn2-unpool{}'.format(i), l, 2)
            l = Conv2D('dsn2-unconv{}'.format(i), l, kernel_shape=3, out_channel=OUT_DIM - 1, padding='SAME', nl=tf.identity)
        l = Conv2D('dsn2-conv-final', l, kernel_shape=3, out_channel=OUT_DIM - 1, padding='SAME', nl=tf.nn.relu)
        l = tf.nn.sigmoid(l, name='boxes')
        outputs.append(l)
        boxes_label = tf.slice(label, [0, 0, 0, 1], [-1, -1, -1, -1])
        boxes_cost = self.IoULoss(l, boxes_label) * 1e-6  # constant for cost balancing

        cost = score_cost + boxes_cost

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.004,
                         regularize_cost('.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)
        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        cost = tf.add_n([cost, wd_cost], name='combined_cost')
        return cost


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = WiderFaceDenseBoxDataset(train_or_test)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, PREFETCH_SIZE, NR_PROC)
    return ds


def get_config():
    # prepare dataset
    step_per_epoch = 128
    dataset_train = get_data('train')
    dataset_test = get_data('test')

    sess_config = get_default_sess_config(0.5)

    nr_gpu = get_nr_gpu()
    lr = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=get_global_step_var(),
        decay_steps=step_per_epoch * 10,
        decay_rate=0.8,
        staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-4),
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
