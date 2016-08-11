#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
import os
import sys
import cv2
import glob
import time
import numpy as np
from data import wider_face_densebox

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

from tensorpack.dataflow.base import DataFlow
from tensorpack.utils import get_rng


class WiderFaceDenseBoxDataset(DataFlow):
    def __init__(self, train_or_test='train', size=-1):
        self.train_or_test = train_or_test
        self.reset_state()
        self.sub_mean = np.array([104.00699, 116.66877, 122.67892], 'float32')
        if size == -1:
            self._size = wider_face_densebox.size_of_dataset(self.train_or_test)
        else:
            self._size = size

    def reset_state(self):
        self.rng = get_rng(self)

    def size(self):
        return self._size

    def get_data(self):
        for idx, (img, label) in enumerate(wider_face_densebox.get_dataset(
                self.train_or_test,
                randomize=True,
        )):
            if idx >= self._size:
                break
            #data = (img.astype('float32') - self.sub_mean) / 255.
            data = img.astype('float32') / 255.
            label = label.transpose(1, 2, 0).astype('float32')
            yield [data, label]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='test', type=str)
    args = parser.parse_args()

    ds = WiderFaceDenseBoxDataset(args.dataset)
    print(ds.size())
    st_time = time.time()
    for i, (data, label) in enumerate(ds.get_data()):
        print "\r%d\t%.4fs" % (i, time.time() - st_time),
        st_time = time.time()
        sys.stdout.flush()
