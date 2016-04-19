#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Tue 19 Apr 2016 06:16:27 PM CST
# Purpose: eval hed
# Mail: hewr2010@gmail.com
import os
import cv2
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str,
                        help="directory name")
    parser.add_argument("--arch", default="architecture.json", type=str)
    parser.add_argument("--weights", default="weights.h5", type=str)
    parser.add_argument("--image_expr", nargs="+", help="e.g. '*.jpg'")
    args = parser.parse_args()
    if len(args.config) > 0:
        args.arch = os.path.join(args.config, args.arch)
        args.weights = os.path.join(args.config, args.weights)
    assert(len(args.image_expr) > 0)
    # get model
    from keras.models import model_from_json
    model = model_from_json(open(args.arch).read())
    model.load_weights(args.weights)
    model.compile('sgd', 'mse')
    # eval
    for img_path in args.image_expr:
        print img_path
        try:
            img = cv2.imread(img_path)
        except:
            continue
        data = np.array([img]).transpose(0, 3, 1, 2)  # NCHW
        pred = model.predict({"input": data}, batch_size=1)["output"]
        cv2.imshow('input', img)
        for idx, res in enumerate(pred):
            cv2.imshow('output%d' % idx, res)
            if chr(cv2.waitKey(0) & 0xFF) == 'q':
                exit()
