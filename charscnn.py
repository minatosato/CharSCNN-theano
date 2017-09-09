#!/usr/local/bin python
#! -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from sklearn.cross_validation import train_test_split
from theano.tensor.nnet import conv
from theano.tensor.signal import pool as  downsample

from layers import ConvolutionalLayer
from layers import EmbedIDLayer
from layers import FullyConnectedLayer
from layers import MaxPoolingLayer
from optimizers import *
from utils import *


class CharSCNN(object):
    def __init__(
        self,
        rng,
        batchsize=100,
        activation=relu
    ):
        
        import char_load
        (num_sent, char_cnt, word_cnt, max_word_len, max_sen_len,\
        k_chr, k_wrd, x_chr, x_wrd, y) = char_load.read("tweets_clean.txt")

        dim_word = 30
        dim_char = 5
        cl_word = 300
        cl_char = 50
        k_word = k_wrd
        k_char = k_chr

        data_train_word,\
        data_test_word,\
        data_train_char,\
        data_test_char,\
        target_train,\
        target_test\
        = train_test_split(x_wrd, x_chr, y, random_state=1234, test_size=0.1)

        x_train_word = theano.shared(np.asarray(data_train_word, dtype='int16'), borrow=True)
        x_train_char = theano.shared(np.asarray(data_train_char, dtype='int16'), borrow=True)
        y_train = theano.shared(np.asarray(target_train, dtype='int8'), borrow=True)
        x_test_word = theano.shared(np.asarray(data_test_word, dtype='int16'), borrow=True)
        x_test_char = theano.shared(np.asarray(data_test_char, dtype='int16'), borrow=True)
        y_test = theano.shared(np.asarray(target_test, dtype='int8'), borrow=True)


        self.n_train_batches = x_train_word.get_value(borrow=True).shape[0] / batchsize
        self.n_test_batches = x_test_word.get_value(borrow=True).shape[0] / batchsize


        
        """symbol definition"""
        index = T.iscalar()
        x_wrd = T.wmatrix('x_wrd')
        x_chr = T.wtensor3('x_chr')
        y = T.bvector('y')
        train = T.iscalar('train')

        """network definition"""
        layer_char_embed_input = x_chr#.reshape((batchsize, max_sen_len, max_word_len))

        layer_char_embed = EmbedIDLayer(
            rng,
            layer_char_embed_input,
            n_input=char_cnt,
            n_output=dim_char
        )

        layer1_input = layer_char_embed.output.reshape(
            (batchsize*max_sen_len, 1, max_word_len, dim_char)
        )

        layer1 = ConvolutionalLayer(
            rng,
            layer1_input,
            filter_shape=(cl_char, 1, k_char, dim_char),# cl_charフィルタ数
            image_shape=(batchsize*max_sen_len, 1, max_word_len, dim_char)
        )

        layer2 = MaxPoolingLayer(
            layer1.output,
            poolsize=(max_word_len-k_char+1, 1)
        )

        layer_word_embed_input = x_wrd #.reshape((batchsize, max_sen_len))

        layer_word_embed = EmbedIDLayer(
            rng,
            layer_word_embed_input,
            n_input=word_cnt,
            n_output=dim_word
        )

        layer3_word_input = layer_word_embed.output.reshape((batchsize, 1, max_sen_len, dim_word))
        layer3_char_input = layer2.output.reshape((batchsize, 1, max_sen_len, cl_char))


        layer3_input = T.concatenate(
            [layer3_word_input,
             layer3_char_input],
            axis=3
        )#.reshape((batchsize, 1, max_sen_len, dim_word+cl_char))


        layer3 = ConvolutionalLayer(
            rng,
            layer3_input,
            filter_shape=(cl_word, 1, k_word, dim_word + cl_char),#1は入力チャネル数
            image_shape=(batchsize, 1, max_sen_len, dim_word + cl_char),
            activation=activation
        )

        layer4 = MaxPoolingLayer(
            layer3.output,
            poolsize=(max_sen_len-k_word+1, 1)
        )

        layer5_input = layer4.output.reshape((batchsize, cl_word))

        layer5 = FullyConnectedLayer(
            rng,
            dropout(rng, layer5_input, train),
            n_input=cl_word,
            n_output=50,
            activation=activation
        )

        layer6_input = layer5.output

        layer6 = FullyConnectedLayer(
            rng,
            dropout(rng, layer6_input, train, p=0.1),
            n_input=50,
            n_output=2,
            activation=None
        )

        result = Result(layer6.output, y)
        loss = result.negative_log_likelihood()
        accuracy = result.accuracy()
        params = layer6.params\
                +layer5.params\
                +layer3.params\
                +layer_word_embed.params\
                +layer1.params\
                +layer_char_embed.params
        updates = RMSprop(learning_rate=0.001, params=params).updates(loss)

        self.train_model = theano.function(
            inputs=[index],
            outputs=[loss, accuracy],
            updates=updates,
            givens={
                x_wrd: x_train_word[index*batchsize: (index+1)*batchsize],
                x_chr: x_train_char[index*batchsize: (index+1)*batchsize],
                y: y_train[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](1)
            }
        )

        self.test_model = theano.function(
            inputs=[index],
            outputs=[loss, accuracy],
            givens={
                x_wrd: x_test_word[index*batchsize: (index+1)*batchsize],
                x_chr: x_test_char[index*batchsize: (index+1)*batchsize],
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
    random_state = 1234
    rng = np.random.RandomState(random_state)
    charscnn = CharSCNN(rng, batchsize=10, activation=relu)
    charscnn.train_and_test(n_epoch=3)






















