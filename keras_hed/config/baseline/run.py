#!/usr/bin/python2
# -*- coding:utf-8 -*-
# Created Time: Sun Feb 21 10:59:34 2016
# Mail: hewr2010@gmail.com
from data import idcard
import numpy as np
import os

from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,\
    Convolution2D, MaxPooling2D, Lambda
from keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_dataset(gnr):
    imgs, labels = [], []
    for img, label in gnr:
        data = img.transpose(2, 0, 1)
        imgs.append(data)
        label = label.reshape((1,) + label.shape)
        label = label.astype("float32") / 255.
        labels.append(label)
    imgs = np.array(imgs, 'uint8')
    labels = np.array(labels, 'float32')
    return imgs, labels


def get_model(in_shape, out_shape):
    def get_conv_network(in_shape, out_shape):
        model = Sequential()
        model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=in_shape))
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(out_shape[0], 3, 3, border_mode='same'))
        return model

    #conv_net = get_conv_network(in_shape, out_shape)
    g = Graph()
    g.add_input(name='input', input_shape=in_shape)
    g.add_node(Lambda(lambda x: x / 255. - 0.5), name='normalize', input='input')
    g.add_node(Convolution2D(out_shape[0], 3, 3, border_mode='same'), name='conv', input='normalize')
    g.add_output(name='output', input='conv')
    g.compile(
        loss={'output': 'mean_squared_error'},
        optimizer=Adam(
            lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        ),
    )
    return g

if __name__ == "__main__":
    # prepare data
    #X_train, y_train = get_dataset(idcard.trainset()) TODO
    X_train, y_train = get_dataset(idcard.testset())
    X_test, y_test = get_dataset(idcard.testset())
    print "data loaded"
    # train model
    model = get_model(X_train.shape[1:], y_train.shape[1:])
    model.fit(
        {'input': X_train, 'output': y_train},
        #validation_data={'input': X_test, 'output': y_test},
        nb_epoch=20,
        batch_size=128,
    )
    # dump model
    open(os.path.join(BASE_DIR, 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join(BASE_DIR, 'weights.h5'), overwrite=True)
    # eval results
    #objective_score = model.evaluate(X_test, y_test, batch_size=16)
    #classes = model.predict_classes(X_test, batch_size=32)
    #proba = model.predict_proba(X_test, batch_size=32)
    from IPython import embed; embed()
