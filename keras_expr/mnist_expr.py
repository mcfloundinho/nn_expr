#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Sun Feb 21 10:59:34 2016
# Purpose: multi-layer perceptrons for mnist dataset
# Mail: hewr2010@gmail.com
from data import mnist
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def one_hot(x, max_len=10):
    a = np.zeros(max_len)
    a[x] = 1
    return a


def get_dataset(gnr):
    imgs, labels = [], []
    for img, label in gnr:
        imgs.append(img.flatten())
        labels.append(one_hot(label))
    imgs = np.array(imgs, 'uint8')
    labels = np.array(labels, 'float32')
    return imgs, labels


def get_model_mlp(in_dim, out_dim):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD

    model = Sequential()
    model.add(Dense(64, input_dim=in_dim, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd)
    return model

if __name__ == "__main__":
    # prepare data
    X_train, y_train = get_dataset(mnist.trainset())
    X_test, y_test = get_dataset(mnist.testset())
    # train model
    model = get_model_mlp(X_train.shape[1], y_train.shape[1])
    model.fit(X_train, y_train,
              nb_epoch=20,
              batch_size=16,
              show_accuracy=True)
    # dump model
    import pickle
    pickle.dump(model, open(
        os.path.join(BASE_DIR, "./trained_models/mnist_mlp.pickle"), "w"))
    # eval results
    objective_score = model.evaluate(X_test, y_test, batch_size=16)
    classes = model.predict_classes(X_test, batch_size=32)
    proba = model.predict_proba(X_test, batch_size=32)
    from IPython import embed; embed()
    exit()
