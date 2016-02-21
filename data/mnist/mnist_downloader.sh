#!/bin/bash
# Created Time: Sun Feb 21 10:27:51 2016
# Author: Wayne Ho
# Mail: hewr2010@gmail.com
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
gunzip train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
