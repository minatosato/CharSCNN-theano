#! -*- coding:utf-8 -*-

from __future__ import print_function
import numpy

def read(inp_file):
    f_in = open(inp_file, 'r')
    lines = f_in.readlines()
    
    words_map = {} #マッピング用
    char_map = {} #マッピング用
    word_cnt = 0 #単語の種類数
    char_cnt = 0 #文字の種類数
    
    k_chr = 3 #文字コンテクストウィンドウ
    k_wrd = 5 #単語コンテクストウィンドウ

    y = [] 
    x_chr = []
    x_wrd = []

    max_word_len, max_sen_len, num_sent = 0, 0, 20000

    for line in lines[:num_sent]:
        words = line[:-1].split()
        tokens = words[1:]
        y.append(int(float(words[0])))
        max_sen_len = max(max_sen_len,len(tokens))
        for token in tokens:
            if token not in words_map:
                words_map[token] = word_cnt
                word_cnt += 1
                max_word_len = max(max_word_len,len(token))
            for i in xrange(len(token)):
                if token[i] not in char_map:
                    char_map[token[i]] = char_cnt
                    char_cnt += 1
    
    for line in lines[:num_sent]:
        words = line[:-1].split()
        tokens = words[1:]
        word_mat = [0] * (max_sen_len+k_wrd-1)
        char_mat = numpy.zeros((max_sen_len+k_wrd-1, max_word_len+k_chr-1))

        for i in xrange(len(tokens)):
            word_mat[(k_wrd/2)+i] = words_map[tokens[i]]
            for j in xrange(len(tokens[i])):
                char_mat[(k_wrd/2)+i][(k_chr/2)+j] = char_map[tokens[i][j]]
        x_chr.append(char_mat)
        x_wrd.append(word_mat)
    max_word_len += k_chr-1
    max_sen_len += k_wrd-1

    # num_sent: 文書の数
    # word_cnt: 単語の種類数
    # char_cnt: 文字の種類数
    # max_sen_len: 文書の最大の長さ
    # max_word_len: 単語の最大の長さ
    # x_chr: 入力となる文字のid列(num_sent*max_sen_len*max_word_len)
    # x_wrd: 入力となる単語のid列(num_sent*max_sen_len)
    # y: 1 or 0 (i.e., positive or negative)

    # print numpy.array(x_wrd).shape
    # print numpy.array(x_chr).shape
    # print num_sent
    # print max_sen_len
    # print max_word_len


    data = (num_sent, char_cnt, word_cnt, max_word_len, max_sen_len,\
            k_chr, k_wrd, x_chr, x_wrd, y)
    return data

read("tweets_clean.txt") 

