#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Sun 17 Apr 2016 12:49:31 PM EDT
# Mail: hewr2010@gmail.com
import os
import cv2
import glob
import random
import itertools

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def generate(prefix, randomize=False):
    paths = glob.glob("%s/*_gt.jpg" % prefix)
    if randomize:
        random.shuffle(paths)
    for gt_path in paths:
        img_path = gt_path[:-7] + "_raw.jpg"
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, 0)
        yield img, gt

def size_of_train():
    return len(glob.glob(os.path.join(BASE_DIR, "train/*_gt.jpg")))

def size_of_test():
    return len(glob.glob(os.path.join(BASE_DIR, "val/*_gt.jpg")))

def trainset(randomize=False):
    return generate(os.path.join(BASE_DIR, "train"), randomize=randomize)

def testset(randomize=False):
    return generate(os.path.join(BASE_DIR, "val"), randomize=randomize)
