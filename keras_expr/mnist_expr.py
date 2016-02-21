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
    Convolution2D, MaxPooling2D, Lambda
from keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mlp", type=str,
                    help="type of model")
parser.add_argument("--learning-rate", default=0.001, type=float)
args = parser.parse_args()


def one_hot(x, max_len=10):
    a = np.zeros(max_len)
    a[x] = 1
    return a


def get_dataset(gnr):
    imgs, labels = [], []
    for img, label in gnr:
        data = img.reshape((1,) + img.shape)  # CHW
        imgs.append(data)
        labels.append(one_hot(label))
    imgs = np.array(imgs, 'uint8')
    labels = np.array(labels, 'float32')
    return imgs, labels


def get_model_mlp(in_shape, out_dim):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=in_shape))
    model.add(Flatten())
    model.add(Dense(64, init='uniform'))
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
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=in_shape))
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
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
    X_train, y_train = get_dataset(mnist.trainset())
    X_test, y_test = get_dataset(mnist.testset())
    # train model
    model = get_model(args.model, X_train.shape[1:], y_train.shape[1])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(
            lr=args.learning_rate,
            beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        ),
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        nb_epoch=20,
        batch_size=128,
        show_accuracy=True,
    )
    # dump model
    open(os.path.join(
        BASE_DIR,
        './trained_models/mnist_{}_architecture.json'.format(args.model)
    ), 'w').write(model.to_json())
    model.save_weights(os.path.join(
        BASE_DIR,
        './trained_models/mnist_{}_weights.h5'.format(args.model)
    ), overwrite=True)
    # eval results
    #objective_score = model.evaluate(X_test, y_test, batch_size=16)
    #classes = model.predict_classes(X_test, batch_size=32)
    #proba = model.predict_proba(X_test, batch_size=32)
    from IPython import embed; embed()
