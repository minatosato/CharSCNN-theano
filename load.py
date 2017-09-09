#! -*- coding:utf-8 -*-


import numpy, theano
import theano.tensor as T

def read(inp_file):
    f_in = open(inp_file, 'r')
    lines = f_in.readlines()
    
    words_map = {}
    word_cnt = 0
    
    k_wrd = 5 #単語コンテクストウィンドウ

    y = [] 
    x_wrd = []

    max_sen_len, num_sent = 0, 20000

    for line in lines[:num_sent]:
        words = line[:-1].split()
        tokens = words[1:]
        y.append(int(float(words[0])))
        max_sen_len = max(max_sen_len,len(tokens))
        for token in tokens:
            if token not in words_map:
                words_map[token] = word_cnt
                word_cnt += 1
    
    for line in lines[:num_sent]:
        words = line[:-1].split()
        tokens = words[1:]
        word_mat = [0] * (max_sen_len+k_wrd-1)

        for i in xrange(len(tokens)):
            word_mat[(k_wrd/2)+i] = words_map[tokens[i]]
        x_wrd.append(word_mat)
    max_sen_len += k_wrd-1

    # num_sent: 文書の数
    # word_cnt: 単語の種類数
    # max_sen_len: 文書の最大の長さ
    # x_wrd: 入力となる単語のid列
    # y: 1 or 0 (i.e., positive or negative)
    data = (num_sent, word_cnt, max_sen_len, k_wrd, x_wrd, y)
    return data
    
# read("tweets_clean.txt") 

