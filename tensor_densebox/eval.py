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
from baseline.data_loader import WiderFaceDenseBoxDataset

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
parser.add_argument('--config', type=str, help='run.py', required=True)
parser.add_argument('--checkpoint', type=str, default='',
                    help='default as "DIR{config}/train_log/checkpoint"')
parser.add_argument('--score_threshold', type=float, default=0.)
parser.add_argument('--nms', action='store_true')
parser.add_argument('--nms_threshold', type=float, default=0.5)
parser.add_argument('--max_height', type=int, default=600)
parser.add_argument('--max_width', type=int, default=800)
args = parser.parse_args()
args.display_shape = (args.max_width, args.max_height)

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
        input_var_names=['input'],
        session_init=SaverRestore(checkpoint_path),
        session_config=get_default_sess_config(0.5),
        output_var_names=['score', 'boxes'],
    )
    predict_func = get_predict_func(pred_config)
    return predict_func


def fit_in_box(img, screen):
    h, w = screen
    r = max(1., max(float(img.shape[0]) / h, float(img.shape[1]) / w))
    img = cv2.resize(img, (int(img.shape[1] / r), int(img.shape[0] / r)))
    return img

if __name__ == '__main__':
    # load user config
    sys.path.append(os.path.dirname(args.config))
    config_module = imp.load_source('_user_config', args.config)
    # assemble prediction function
    predict_func = assemble_func(config_module, args.checkpoint)

    ds = WiderFaceDenseBoxDataset('test')
    for data, _ in ds.get_data():
        img = (data.copy() * 255).astype('uint8')
        data = data.reshape((1,) + data.shape)
        with ScopedTimer() as timer:
            pred_res = predict_func([data])
            heatmap = pred_res[0][0, :, :, 0]
            print("heatmap l2loss: %.6f" % (((_[:, :, 0] - heatmap) ** 2).mean()))
            print("heatmap min(%.6f) max(%.6f)" % (heatmap.min(), heatmap.max()))
            heatmap[heatmap < args.score_threshold] = 0
            label = pred_res[1][0].transpose(2, 0, 1)
        print 'time passed:', timer.interval
        # heatmap
        merged_img = (img.astype('float32') * np.maximum(0.2, heatmap)\
                      .reshape(heatmap.shape + (1,))).astype('uint8')
        cv2.imshow('merge', fit_in_box(merged_img, args.display_shape))
        cv2.imshow('heatmap', fit_in_box(heatmap, args.display_shape))
        # boxes
        h, w = img.shape[:2]
        x_mat = np.tile(np.arange(w).reshape(1, w), (h, 1)).astype('float32')
        y_mat = np.tile(np.arange(h).reshape(h, 1), (1, w)).astype('float32')
        box_mat = [x_mat - label[0], y_mat - label[1], label[0] + label[2], label[1] + label[3]]
        mask = heatmap > 0
        bot_mat = [mat[mask] for mat in box_mat]
        boxes = list(zip(*bot_mat))
        boxes = [list(map(int, box)) for box in boxes]
        # nms
        if args.nms:
            with ScopedTimer() as timer:
                boxes = nms.perform_iou_nms_avgbox(boxes, args.nms_threshold)
            print 'nms time: %.2s' % timer.interval
        for x, y, w, h in boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.imshow('boxes', fit_in_box(img, args.display_shape))
        if chr(cv2.waitKey(0) & 0xFF) == 'q':
            exit()
