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


def data_generator(gnr_caller, batch_size=1):
    gnr = gnr_caller()
    while True:
        imgs, labels = [], []
        while len(imgs) < batch_size:
            try:
                img, label = gnr.next()
            except StopIteration:
                gnr = gnr_caller()
                continue
            data = img.transpose(2, 0, 1)
            imgs.append(data)
            label = label.reshape((1,) + label.shape)
            label = label.astype("float32") / 255.
            labels.append(label)
        imgs = np.array(imgs, 'uint8')
        labels = np.array(labels, 'float32')
        yield {'input': imgs, 'output': labels}


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
    # prepare data generator
    train_dataset = data_generator(lambda: idcard.trainset(randomize=True))
    test_dataset = data_generator(lambda: idcard.testset(randomize=True))
    # train model
    datapoint = train_dataset.next()
    model = get_model(datapoint["input"].shape[1:], datapoint["output"].shape[1:])
    model.fit_generator(
        train_dataset,
        samples_per_epoch=1024,
        nb_epoch=100000000,
        validation_data=test_dataset,
        nb_val_samples=128,
    )
    # dump model
    open(os.path.join(BASE_DIR, 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join(BASE_DIR, 'weights.h5'), overwrite=True)
