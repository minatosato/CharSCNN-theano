#!/usr/local/bin python
#! -*- coding: utf-8 -*-

from __future__ import print_function

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool as downsample
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

def relu(x):
    return T.maximum(0, x)
def sigmoid(x):
    return T.nnet.sigmoid(x)
def tanh(x):
    return T.tanh(x)

def dropout(rng, x, train, p=0.5):
    masked_x = None
    if p > 0.0 and p < 1.0:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(
            n=1,
            p=1.0-p,
            size=x.shape,
            dtype=theano.config.floatX
        )
        # mask *= 1.0 / (1.0 - p)
        masked_x = x * mask
    else:
        masked_x = x
    return T.switch(T.neq(train, 0), masked_x, x*(1.0-p))

def get_corrupted_input(rng, x, train, corruption_level=0.3):
    masked_x = None
    if corruption_level > 0.0 and corruption_level < 1.0:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(
            n=1,
            p=1.0-corruption_level,
            size=x.shape,
            dtype=theano.config.floatX
        )
        masked_x = x * mask
    return T.switch(T.neq(train, 0), masked_x, x)

def max_pool_2d(x, poolsize=(2,2)):
    pooled_out = downsample.pool_2d(
        input = x,
        ws = poolsize,
        ignore_border = True
    )
    return pooled_out

def embed_id(sentences=None, n_vocab=None, k_wrd=None):
    if sentences is None or n_vocab is None or k_wrd is None:
        return NotImplementedError()

    tmp = sentences.get_value(borrow=True)
    max_sent_len = len(tmp[0])
    x_wrd = []
    for sentence in tmp:
        word_mat = np.array([[0]*n_vocab]*(max_sent_len+k_wrd-1), dtype='int8')

        i = 0
        for word in sentence:
            word_mat[(k_wrd/2)+i][word] = 1
            i += 1

        x_wrd.append(word_mat)
    return theano.shared(x_wrd, borrow=False)



class Result(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def negative_log_likelihood(self):
        self.prob_of_y_given_x = T.nnet.softmax(self.x)
        return -T.mean(T.log(self.prob_of_y_given_x)[T.arange(self.y.shape[0]), self.y])

    def cross_entropy(self):
        self.prob_of_y_given_x = T.nnet.softmax(self.x)
        return T.mean(T.nnet.categorical_crossentropy(self.prob_of_y_given_x, self.y))

    def mean_squared_error(self):
        return T.mean((self.x - self.y) ** 2)

    def errors(self):
        if self.y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', self.y.type, 'y_pred', self.y_pred.type))

        if self.y.dtype.startswith('int'):
            self.prob_of_y_given_x = T.nnet.softmax(self.x)
            self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
            return T.mean(T.neq(self.y_pred, self.y))
        else:
            return NotImplementedError()

    def accuracy(self):
        if self.y.dtype.startswith('int'):
            self.prob_of_y_given_x = T.nnet.softmax(self.x)
            self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
            return T.mean(T.eq(self.y_pred, self.y))
        else:
            return NotImplementedError()

def load_data(random_state=0):
    print('fetch MNIST dataset')
    mnist = fetch_mldata('MNIST original')
    mnist.data   = mnist.data.astype(np.float32)
    mnist.data  /= 255
    mnist.target = mnist.target.astype(np.int32)

    data_train,\
    data_test,\
    target_train,\
    target_test\
    = train_test_split(mnist.data, mnist.target, random_state=random_state)

    def shared_data(x,y):
        shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)

        return shared_x, T.cast(shared_y, 'int32')

    data_train, target_train = shared_data(data_train, target_train)
    data_test, target_test = shared_data(data_test, target_test)

    return ([data_train, data_test], [target_train, target_test])

def load_livedoor_news_corpus(random_state=0, test_size=0.1):
    import six.moves.cPickle as pickle
    data_, target_ = pickle.load(open('dataset', 'rb'))

    data_train,\
    data_test,\
    target_train,\
    target_test\
    = train_test_split(data_, target_, random_state=random_state, test_size=test_size)

    def shared_data(x,y):
        shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)

        return shared_x, T.cast(shared_y, 'int32')

    data_train, target_train = shared_data(data_train, target_train)
    data_test, target_test = shared_data(data_test, target_test)

    return ([data_train, data_test], [target_train, target_test])


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX), 
        name=name, 
        borrow=True
    )

















