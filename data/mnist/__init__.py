#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Sun Feb 21 10:55:29 2016
# Purpose: mnist data provider
# Mail: hewr2010@gmail.com
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def generate(images_fn, labels_fn):
    from parse import parse_images, parse_labels
    images_gnr = parse_images(images_fn)
    labels_gnr = parse_labels(labels_fn)
    import itertools
    return itertools.izip(images_gnr, labels_gnr)


def trainset():
    return generate(
        os.path.join(BASE_DIR, "./train-images-idx3-ubyte"),
        os.path.join(BASE_DIR, "./train-labels-idx1-ubyte"),
    )


def testset():
    return generate(
        os.path.join(BASE_DIR, "./t10k-images-idx3-ubyte"),
        os.path.join(BASE_DIR, "./t10k-labels-idx1-ubyte"),
    )
