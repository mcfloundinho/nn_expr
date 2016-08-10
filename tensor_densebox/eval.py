#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
import os
import sys
import cv2
import imp
import time
import numpy as np
from baseline.data_loader import FlyingChairDataset

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
parser.add_argument('--config', type=str, help='run.py', required=True)
parser.add_argument('--checkpoint', type=str, default='',
                    help='default as "DIR{config}/train_log/checkpoint"')
args = parser.parse_args()

import tensorflow as tf
from tensorpack.train import TrainConfig, QueueInputTrainer
from tensorpack.predict import PredictConfig, get_predict_func
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.utils import *
from tensorpack.tfutils import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import *

logger.set_logger_dir(os.path.join(BASE_DIR, '.log'), action='d')

if args.checkpoint == "":
    args.checkpoint = os.path.join(os.path.dirname(args.config),
                                   "train_log/checkpoint")
    logger.info('checkpoint set to "%s"' % args.checkpoint)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ScopedTimer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def assemble_func(config_module, checkpoint_path):
    model = config_module.Model()
    pred_config = PredictConfig(
        model=model,
        input_data_mapping=[0, 1],
        session_init=SaverRestore(checkpoint_path),
        output_var_names=['pred'],
    )
    predict_func = get_predict_func(pred_config)
    return predict_func

if __name__ == '__main__':
    # load user config
    sys.path.append(os.path.dirname(args.config))
    config_module = imp.load_source('_user_config', args.config)
    # assemble prediction function
    predict_func = assemble_func(config_module, args.checkpoint)

    ds = FlyingChairDataset('test')
    for ldata, rdata, label in ds.get_data():
        ldata = ldata.reshape((1,) + ldata.shape)
        rdata = rdata.reshape((1,) + rdata.shape)
        with ScopedTimer() as timer:
            pred_res = predict_func([ldata, rdata])
            x, y, w, h = pred_res[0][0]
        print 'time passed:', timer.interval
        limg = (ldata[0] * 255).astype('uint8')
        rimg = (rdata[0] * 255).astype('uint8')
        cv2.rectangle(rimg, (int(x * rimg.shape[1]), int(y * rimg.shape[0])),
                      (int((w + x) * rimg.shape[1]), int((h + y) * rimg.shape[0])),
                       (255, 0, 0));
        import cv2
        cv2.imshow('left', limg)
        cv2.imshow('right', rimg)
        if chr(cv2.waitKey(0) & 0xFF) == 'q':
            exit()
