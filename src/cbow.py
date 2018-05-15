# -*- coding: utf8 -*-

import math
import argparse
import collections
import numpy as np
import tensorflow as tf

class CBOW(object):
    def __init__(self, corpus, vocabulary_size, word_embedding_size, learning_rate, window_size, batch_size, epoches, save_file):
        self.cur = 0
        self.epoches = epoches
        self.save_file = save_file
        self.batch_size = batch_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.vocabulary_size = vocabulary_size
        self.word_embedding_size = word_embedding_size
        with open(corpus, 'r', encoding = 'utf8') as f:
            words = f.read().split()
        self.word_number = len(words)
        self.build_dataset(words)
        del words
    
    def build_dataset(self, words):
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        cont, inx, self.index, self.dictionary = 0, 0, list(), dict()
        for word, _ in self.count:
            if word not in self.dictionary:
                self.dictionary[word] = inx
                inx += 1
        for word in words:
            if word not in self.dictionary:
                cont += 1
                self.index.append(self.dictionary['UNK'])
            else:
                self.index.append(self.dictionary[word])
        self.count[0][1] = cont

    def generate_batch(self):
        x, y = [[] for i in range(self.batch_size)], [0 for i in range(self.batch_size)]
        for i in range(self.batch_size):
            if self.cur + self.window_size >= self.word_number:
                self.cur = 0
            x[i] = self.index[self.cur: self.cur + self.window_size // 2] + self.index[self.cur + self.window_size // 2 + 1: self.cur + self.window_size]
            y[i] = self.index[self.window_size // 2]
        return x, y

    def train(self):
        with tf.name_scope('input_layer'):
            C = tf.Variable(tf.random_uniform([self.vocabulary_size, self.word_embedding_size], -1.0, 1.0))
            x = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.window_size - 1])
            y = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
            inputs = tf.nn.embedding_lookup(C, x)
            labels = tf.one_hot(y, self.vocabulary_size)
        with tf.name_scope('projection_layer'):
            h = tf.reduce_mean(inputs, 1) # shape: batch_size * word_embedding_size
        with tf.name_scope('output_layer'):
            V = tf.Variable(tf.random_normal(shape = [self.word_embedding_size, self.vocabulary_size], mean = 0.0, stddev = 0.1))
            output = tf.matmul(h, V)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output))
            train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoches):
                _x, _y = self.generate_batch()
                sess.run(train, feed_dict = {x: _x, y: _y})
                if epoch % 100 == 99:
                    print ("Epoch #%d, loss:" % (epoch + 1), sess.run(loss, feed_dict = {x: _x, y: _y}))
            self.word_embedding = sess.run(C)

    def save(self):
        with open(self.save_file, 'w', encoding = 'utf8') as f:
            for word in self.dictionary:
                f.write(word + ": " + str([x for x in self.word_embedding[self.dictionary[word]]]) + "\n")
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--corpus', type=str, help="语料文件路径", required=True)
    parser.add_argument('-VS', '--vocabulary-size', type=int, help="字典大小", default=10000)
    parser.add_argument('-WES', '--word-embedding-size', type=int, help="词向量的大小", default=20)
    parser.add_argument('-LR', '--learning-rate', type=float, help="学习速率", default=0.01)
    parser.add_argument('-WS', '--window-size', type=int, help="目标词所需的上文词数", default=15)
    parser.add_argument('-BS', '--batch-size', type=int, help="批量生成数据的大小", default=50)
    parser.add_argument('-E', '--epoches', type=int, help="迭代次数", default=10000)
    parser.add_argument('-SF', '--save-file', type=str, help="生成的词向量保存的位置", default="data/cbow.txt")
    args = parser.parse_args()

    cbow = CBOW(args.corpus, args.vocabulary_size, args.word_embedding_size, args.learning_rate, args.window_size, args.batch_size, args.epoches, args.save_file)
    cbow.train()
    cbow.save()
