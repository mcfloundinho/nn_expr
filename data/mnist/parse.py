#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Sun Feb 21 10:34:02 2016
# Purpose: parse mnist data
# Mail: hewr2010@gmail.com
import numpy as np

# bytes_to_int(file.read(LENGTH))
bytes_to_int = lambda s: reduce(lambda x, y: x * 256 + y, map(ord, list(s)))


def parse_images(fn):
    fin = open(fn)
    assert bytes_to_int(fin.read(4)) == 2051  # verification
    nr_inst = bytes_to_int(fin.read(4))
    nr_row, nr_col = bytes_to_int(fin.read(4)), bytes_to_int(fin.read(4))
    for _ in xrange(nr_inst):
        yield np.array(map(ord, list(fin.read(nr_row * nr_col))))\
            .reshape(nr_row, nr_col).astype('uint8')


def parse_labels(fn):
    fin = open(fn)
    assert bytes_to_int(fin.read(4)) == 2049  # verification
    nr_inst = bytes_to_int(fin.read(4))
    for _ in xrange(nr_inst):
        yield bytes_to_int(fin.read(1))

if __name__ == "__main__":
    import cv2
    import itertools
    images_gnr = parse_images("./train-images-idx3-ubyte")
    labels_gnr = parse_labels("./train-labels-idx1-ubyte")
    for idx, (img, label) in enumerate(itertools.izip(images_gnr, labels_gnr)):
        print "{}\t{}".format(idx, label)
        cv2.imshow('', img)
        if chr(cv2.waitKey(0) & 0xFF) == 'q':
            exit()
