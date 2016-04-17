#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Sun 17 Apr 2016 12:49:31 PM EDT
# Mail: hewr2010@gmail.com
import os
import cv2
import glob
import itertools

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def generate(prefix):
    for gt_path in glob.glob("%s/*_gt.jpg" % prefix):
        img_path = gt_path[:-7] + ".jpg"
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, 0)
        yield img, gt


def trainset():
    return generate(os.path.join(BASE_DIR, "train"))


def testset():
    return generate(os.path.join(BASE_DIR, "val"))
