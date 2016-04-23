#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
import os
import cv2
import glob

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

from tensorpack.dataflow.base import DataFlow
from tensorpack.utils import get_rng


class IDCardDataset(DataFlow):
    def __init__(self, prefix):
        paths = glob.glob(os.path.join(BASE_DIR, 'data/idcard/%s/*_raw.jpg' % prefix))
        self.path_pairs = [(s, s[:-8] + "_gt.jpg") for s in paths]
        self.reset_state()

    def reset_state(self):
        self.rng = get_rng(self)

    def size(self):
        return len(self.path_pairs)

    def get_data(self):
        for i in self.rng.randint(len(self.path_pairs), size=self.size()):
            img_path, gt_path = self.path_pairs[i]
            img = cv2.imread(img_path)
            img = img.astype('float32') / 255.
            label = cv2.imread(gt_path, 0)
            label = label.astype('float32') / 255.
            label = label.reshape(label.shape[:2] + (1,))
            yield [img, label]


def get_dataset(train_or_test):
    if train_or_test == 'train':
        return IDCardDataset('train')
    elif train_or_test == "test":
        return IDCardDataset('val')
    else:
        raise NotImplementedError

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='test', type=str)
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    for img, labels in ds.get_data():
        import cv2
        cv2.imshow('img', img)
        for idx, label in enumerate(labels.transpose(2, 0, 1)):
            cv2.imshow('label%d' % idx, label)
        if chr(cv2.waitKey(0) & 0xFF) == 'q':
            exit()
