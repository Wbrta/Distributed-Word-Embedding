# -*- coding: utf8 -*-

import math
import random
import argparse
import collections
import numpy as np
import tensorflow as tf

class SkipGram(object):
    def __init__(self, corpus, vocabulary_size, batch_size, window_size, word_embedding_size, learning_rate, epoches, save_file):
        self.cur = 0
        self.epoches = epoches
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
        self.save_file = save_file

    def build_dataset(self, words):
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        cont, inx, self.index, self.dictionary = 0, 0, list(), dict()
        for word, _ in self.count:
            if word not in self.dictionary:
                self.dictionary[word] = inx
                inx += 1
        for word in words:
            inx = self.dictionary.get(word, 0)
            if inx == 0:
                cont += 1
            self.index.append(inx)
        self.count[0][1] = cont
    
    def generate_batch(self):
        def generate_ones(batches, length):
            y = [[0 for j in range(length)] for i in range(len(batches))]
            for i in range(len(batches)):
                for index in batches[i]:
                    y[i][index] = 1
            return y

        x, tmp = [0 for i in range(self.batch_size)], [[] for i in range(self.batch_size)]
        for i in range(self.batch_size):
            if self.cur + self.window_size >= self.word_number:
                self.cur = 0
            x[i] = self.index[self.cur + self.window_size // 2]
            tmp[i].extend(self.index[self.cur: self.cur + self.window_size // 2])
            tmp[i].extend(self.index[self.cur + self.window_size // 2 + 1: self.cur + self.window_size])
            self.cur += random.randint(1, self.batch_size)
        return x, generate_ones(tmp, self.vocabulary_size)
 
    def train(self):
        with tf.name_scope('input_layer'):
            inputs = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
            labels = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.vocabulary_size])
        with tf.name_scope('embedding_layer'):
            C = tf.Variable(tf.random_normal([self.vocabulary_size, self.word_embedding_size]))
            e = tf.nn.embedding_lookup(C, inputs)
        with tf.name_scope('output_layer'):
            V = tf.Variable(tf.random_normal(shape = [self.word_embedding_size, self.vocabulary_size], mean = 0.0, stddev = 1 / math.sqrt(self.word_embedding_size)))
            outputs = tf.matmul(e, V)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = outputs, labels = labels))
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoches):
                x, y = self.generate_batch()
                sess.run(optimizer, feed_dict = {inputs: x, labels: y})
                if epoch % 100 == 99:
                    print ("Epoch #%d, loss:" % (epoch + 1), sess.run(loss, feed_dict = {inputs: x, labels: y}))
            self.word_embedding = sess.run(C)

    def save(self):
        with open(self.save_file, 'w', encoding = 'utf8') as f:
            for word in self.dictionary:
                f.write(word + ": " + str([x for x in self.word_embedding[self.dictionary[word]]]) + "\n")
                f.flush()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--corpus', type = str, help = "语料文件路径", required = True)
    parser.add_argument('-VS', '--vocabulary_size', type = int, help = "字典大小", default = 10000)
    parser.add_argument('-BS', '--batch-size', type = int, help = "批量生成数据的大小", default = 25)
    parser.add_argument('-WS', '--window-size', type = int, help = "目标词所需的上下文词数", default = 10)
    parser.add_argument('-WES', '--word-embedding-size', type = int, help = "词向量的大小", default = 300)
    parser.add_argument('-LR', '--learning-rate', type = float, help = "学习速率", default = 0.01)
    parser.add_argument('-E', '--epoches', type = int, help="迭代次数", default = 10000)
    parser.add_argument('-SF', '--save-file', type = str, help="生成的词向量保存的位置", default="data/skip_gram.txt")
    args = parser.parse_args()
 
    sg = SkipGram(args.corpus, args.vocabulary_size, args.batch_size, args.window_size, args.word_embedding_size, args.learning_rate, args.epoches, args.save_file)
    sg.train()
    sg.save()