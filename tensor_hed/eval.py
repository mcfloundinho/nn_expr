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

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

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
        input_data_mapping=[0],
        session_init=SaverRestore(checkpoint_path),
        output_var_names=['upsample_final_sigmoid/output:0']   # output:0 is the probability distribution
    )
    predict_func = get_predict_func(pred_config)
    return predict_func

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--threshold', type=float, default=0.,
                        help='for heatmap')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--config', type=str, help='run.py', required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    logger.set_logger_dir(os.path.join(BASE_DIR, '.log'), action='d')

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # load user config
    sys.path.append(os.path.dirname(args.config))
    config_module = imp.load_source('_user_config', args.config)
    # assemble prediction function
    predict_func = assemble_func(config_module, args.checkpoint)

    for img_path in args.images:
        print img_path
        img = cv2.imread(img_path)
        if img is None:
            print 'fail loading'
            continue
        img = cv2.resize(img, config_module.IMAGE_SHAPE[::-1])
        if config_module.NR_CHANNEL == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #else:
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = np.reshape(img, (1,) + img.shape).astype('float32') / 255.
        with ScopedTimer() as timer:
            pred_res = predict_func([data])
        print 'time passed:', timer.interval
        outputs = pred_res[0][0].transpose(2, 0, 1)
        cv2.imshow('input', img)
        for chn, output in enumerate(outputs):
            out_img = output.copy()
            print 'channel%d' % chn, out_img.min(), out_img.max()
            out_img[out_img < args.threshold] = 0
            cv2.imshow('output%d' % chn, out_img)
            if chr(cv2.waitKey(0) & 0xFF) == 'q':
                exit()
