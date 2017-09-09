#!/usr/local/bin python
#! -*- coding: utf-8 -*-

from __future__ import print_function

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool as downsample
import numpy as np
from sklearn.cross_validation import train_test_split

from layers import EmbedIDLayer
from layers import FullyConnectedLayer
from layers import ConvolutionalLayer
from layers import MaxPoolingLayer
from utils import *
from optimizers import *

# 31566 100


class SCNN(object):
    def __init__(
        self,
        rng,
        batchsize=100,
        activation=tanh
    ):
        
        import load
        (num_sent, word_cnt, max_sen_len, k_wrd, x_wrd, y) \
        = load.read("tweets_clean.txt")


        dim_word = 100
        cl_word = 300
        k_wrd = 5
        vocab_size = word_cnt
        n_hidden = 300

        data_train,\
        data_test,\
        target_train,\
        target_test\
        = train_test_split(x_wrd, y, random_state=1234, test_size=0.1)

        x_train = theano.shared(np.asarray(data_train, dtype='int16'), borrow=True)
        y_train = theano.shared(np.asarray(target_train, dtype='int32'), borrow=True)
        x_test = theano.shared(np.asarray(data_test, dtype='int16'), borrow=True)
        y_test = theano.shared(np.asarray(target_test, dtype='int32'), borrow=True)

        self.n_train_batches = x_train.get_value(borrow=True).shape[0] / batchsize
        self.n_test_batches = x_test.get_value(borrow=True).shape[0] / batchsize


        
        """symbol definition"""
        index = T.iscalar()
        x = T.wmatrix('x')
        y = T.ivector('y')
        train = T.iscalar('train')


        layer_embed_input = x#.reshape((batchsize, max_sen_len))

        layer_embed = EmbedIDLayer(
            rng,
            layer_embed_input,
            n_input=vocab_size,
            n_output=dim_word,
        )

        layer1_input = layer_embed.output.reshape((batchsize, 1, max_sen_len, dim_word))

        layer1 = ConvolutionalLayer(
            rng,
            layer1_input,
            filter_shape=(cl_word, 1, k_wrd, dim_word),#1は入力チャネル数
            image_shape=(batchsize, 1, max_sen_len, dim_word),
            activation=activation
        )

        layer2 = MaxPoolingLayer(
            layer1.output,
            poolsize=(max_sen_len-k_wrd+1, 1)
        )

        layer3_input = layer2.output.reshape((batchsize, cl_word))

        layer3 = FullyConnectedLayer(
            rng,
            dropout(rng, layer3_input, train),
            n_input=cl_word,
            n_output=n_hidden,
            activation=activation
        )

        layer4 = FullyConnectedLayer(
            rng,
            dropout(rng, layer3.output, train),
            n_input=n_hidden,
            n_output=2,
            activation=None
        )

        result = Result(layer4.output, y)
        # loss = result.negative_log_likelihood()
        loss = result.cross_entropy()
        accuracy = result.accuracy()
        params = layer4.params + layer3.params + layer1.params + layer_embed.params
        # updates = AdaDelta(params=params).updates(loss)
        updates = RMSprop(learning_rate=0.001, params=params).updates(loss)
        

        self.train_model = theano.function(
            inputs=[index],
            outputs=[loss, accuracy],
            updates=updates,
            givens={
                x: x_train[index*batchsize: (index+1)*batchsize],
                y: y_train[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](1)
            }
        )

        self.test_model = theano.function(
            inputs=[index],
            outputs=[loss, accuracy],
            givens={
                x: x_test[index*batchsize: (index+1)*batchsize],
                y: y_test[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](0)
            }
        )


    def train_and_test(self, n_epoch=100):
        epoch = 0
        accuracies = []
        while epoch < n_epoch:
            epoch += 1
            sum_loss = 0
            sum_accuracy = 0
            for batch_index in xrange(self.n_train_batches):
                batch_loss, batch_accuracy = self.train_model(batch_index)
                sum_loss = 0
                sum_accuracy = 0
                for batch_index in xrange(self.n_test_batches):
                    batch_loss, batch_accuracy = self.test_model(batch_index)
                    sum_loss += batch_loss
                    sum_accuracy += batch_accuracy
                loss = sum_loss / self.n_test_batches
                accuracy = sum_accuracy / self.n_test_batches
                accuracies.append(accuracy)

                print('epoch: {}, test mean loss={}, test accuracy={}'.format(epoch, loss, accuracy))
                print('')
        return accuracies




if __name__ == '__main__':
    tmp = []
    seed = 10
    for _ in xrange(seed):
        rng = np.random.RandomState(_)
        scnn = SCNN(rng, batchsize=100, activation=relu)
        accuracies = scnn.train_and_test(n_epoch=3)
        tmp.append(accuracies)
    tmp = np.array(tmp)
    mean = [np.mean(tmp[:,i]) for i in range(len(tmp[0]))]
    mean = np.array(mean)
    np.savetxt('scnn.csv', mean, delimiter=',')






















