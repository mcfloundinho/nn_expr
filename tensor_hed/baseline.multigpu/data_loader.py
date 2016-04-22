#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Thu 21 Apr 2016 11:28:50 PM CST
# Mail: hewr2010@gmail.com
from data import idcard
from tensorpack.dataflow.base import DataFlow
from tensorpack.utils import get_rng


class GeneratorWrapperDataset(DataFlow):
    def __init__(self, gnr_constructor, _size):
        """
        :param gnr_constructor: return a generator which yields (image, label)
        :param _size: how many data points in a row
        """
        self.gnr_constructor = gnr_constructor
        self._size = _size
        self.reset_state()

    def reset_state(self):
        self.rng = get_rng(self)
        self.gnr = self.gnr_constructor()

    def size(self):
        return self._size

    def get_data(self):
        count = 0
        while count < self._size:
            try:
                img, label = self.gnr.next()
            except StopIteration:
                self.reset_state()
                continue
            count += 1
            img = img.astype('float32') / 255.
            label = label.astype('float32') / 255.
            label = label.reshape(label.shape[:2] + (1,))
            yield [img, label]


def get_dataset(train_or_test):
    if train_or_test == 'train':
        return GeneratorWrapperDataset(
            lambda: idcard.trainset(randomize=True),
            idcard.size_of_train(),
        )
    elif train_or_test == "test":
        return GeneratorWrapperDataset(
            lambda: idcard.testset(randomize=True),
            idcard.size_of_test(),
        )
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
        for idx, label in enumerate(labels):
            cv2.imshow('label%d' % idx, label)
        if chr(cv2.waitKey(0) & 0xFF) == 'q':
            exit()
