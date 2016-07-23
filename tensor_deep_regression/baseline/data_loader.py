#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
import os
import cv2
import glob
import numpy as np
from data import synth_flying_chairs

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

from tensorpack.dataflow.base import DataFlow
from tensorpack.utils import get_rng


class FlyingChairDataset(DataFlow):
    def __init__(self, train_or_test='train'):
        self.train_or_test = train_or_test
        self.reset_state()

    def reset_state(self):
        self.rng = get_rng(self)
        self.ds = synth_flying_chairs.get_dataset(self.train_or_test, randomize=True)

    def size(self):
        return synth_flying_chairs.size_of_dataset(self.train_or_test)

    def get_data(self):
        for limg, rimg, gt in self.ds:
            ldata = limg.astype('float32') / 255.
            rdata = rimg.astype('float32') / 255.
            label = np.array(gt, 'float32')
            yield [ldata, rdata, label]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='test', type=str)
    args = parser.parse_args()

    ds = FlyingChairDataset(args.dataset)
    for ldata, rdata, label in ds.get_data():
        pass
