#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Sun Feb 21 10:59:34 2016
# Purpose: multi-layer perceptrons for mnist dataset
# Mail: hewr2010@gmail.com
from data import mnist
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,\
    Convolution2D, MaxPooling2D
from keras.optimizers import SGD

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cnn", type=str,
                    help="type of model")
args = parser.parse_args()


def one_hot(x, max_len=10):
    a = np.zeros(max_len)
    a[x] = 1
    return a


def get_dataset(gnr, is_flatten=False):
    imgs, labels = [], []
    for img, label in gnr:
        data = img.flatten() if is_flatten else img.reshape((1,) + img.shape)
        imgs.append(data)
        labels.append(one_hot(label))
    imgs = np.array(imgs, 'uint8')
    labels = np.array(labels, 'float32')
    return imgs, labels


def get_model_mlp(in_dim, out_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=in_dim, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim, init='uniform'))
    model.add(Activation('softmax'))
    return model


def get_model_cnn(in_shape, out_dim):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(out_dim))
    model.add(Activation('softmax'))

    return model


def get_model(model_type, *args, **kwargs):
    if model_type == "cnn":
        return get_model_cnn(*args, **kwargs)
    elif model_type == "mlp":
        return get_model_mlp(*args, **kwargs)
    else:
        raise NotImplementedError(model_type)

if __name__ == "__main__":
    # prepare data
    is_flatten = args.model == "mlp"
    X_train, y_train = get_dataset(mnist.trainset(), is_flatten)
    X_test, y_test = get_dataset(mnist.testset(), is_flatten)
    # train model
    model = get_model(
        args.model,
        X_train.shape[1] if is_flatten else X_train.shape[1:],
        y_train.shape[1],
    )
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd)
    model.fit(X_train, y_train,
              nb_epoch=20,
              batch_size=16,
              show_accuracy=True)
    # dump model
    import pickle
    pickle.dump(
        model,
        open(os.path.join(
            BASE_DIR,
            "./trained_models/mnist_{}.pickle".format(args.model)
        ), "w"),
    )
    # eval results
    objective_score = model.evaluate(X_test, y_test, batch_size=16)
    classes = model.predict_classes(X_test, batch_size=32)
    proba = model.predict_proba(X_test, batch_size=32)
    from IPython import embed; embed()
    exit()
