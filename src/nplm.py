# -*- coding: utf8 -*-

import math
import argparse
import collections
import numpy as np
import tensorflow as tf

class NPLM(object):
    def __init__(self, filename, window_size, hidden_size, word_embedding_size, learning_rate, step, batch_size):
        # 字典相关
        self.index = None
        self.count = None
        self.dictionary = None
        self.rdictionary = None
        # 模型参数相关
        self.window_size = 0
        self.hidden_size = 0
        self.learning_rate = 0
        self.vocabulary_size = 0
        self.word_embedding_size = 0
        # 批量数据
        self.cur = 0
        self.step = 0
        self.batch_size = 0
        # 模型
        self.C = None
        self.H = None
        self.b1 = None
        self.U = None
        self.b2 = None

        with open(filename, "r") as file:
            text = file.read()
        words = text.split()
        self.build_dataset(words)
        del text, words
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.vocabulary_size = len(self.dictionary)
        self.word_embedding_size = word_embedding_size
        self.step = step
        self.batch_size = batch_size

    def sum_target(self, y, index):
        res = tf.zeros([1])
        for inx in index:
            res = tf.add(res, y[inx])
        return res[0]

    def train(self):
        words, target_word = self.generate_batch()
        words = words.reshape([self.window_size * self.batch_size])
        with tf.name_scope('input_layer'):
            self.C = tf.Variable(tf.random_uniform([self.vocabulary_size, self.word_embedding_size], -1.0, 1.0))
            e = tf.nn.embedding_lookup(self.C, words)        # shape = [window_size * batch_size, word_embedding_size]
            e = tf.reshape(e, shape = [self.batch_size, self.window_size * self.word_embedding_size])
        with tf.name_scope('hidden_layer'):
            self.H = tf.Variable(tf.truncated_normal(shape = [self.window_size * self.word_embedding_size, self.hidden_size], stddev = 1.0 / math.sqrt(self.word_embedding_size)))
            self.b1 = tf.Variable(tf.constant(0.1, shape = [self.batch_size, self.hidden_size]))
            h = tf.tanh(tf.add(tf.matmul(e, self.H), self.b1))  # shape = [batch_size, hidden_size]
        with tf.name_scope('output_layer'):
            self.U = tf.Variable(tf.truncated_normal(shape = [self.hidden_size, self.vocabulary_size], stddev = 1.0 / math.sqrt(self.word_embedding_size)))
            self.b2 = tf.Variable(tf.constant(0.1, shape = [self.batch_size, self.vocabulary_size]))
            y_ = tf.nn.softmax(tf.add(tf.matmul(h, self.U), self.b2))   # shape = [batch_size, vocabulary_size]
        with tf.name_scope('index'):
            for i in range(self.batch_size):
                
            
        with tf.name_scope('loss_function'):
            regulation = tf.nn.l2_loss(self.H) + tf.nn.l2_loss(self.U)
            penalized_log_likelihood = -(tf.reduce_mean(tf.log()) + regulation)
            train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(penalized_log_likelihood)
        return train, penalized_log_likelihood

    def test(self):
        train, penalized_log_likelihood = self.train()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(self.step):
                sess.run(train)
                print ("penalized_log_likelihood:", sess.run(penalized_log_likelihood))

    def save(self):
        pass

    def build_dataset(self, words):
        """
        创建单词的单词表

        Args:
            words: 文件中的所有词，type: list[string]
        Return:
            index: 所有在文件中出现过的词在词典中的位置，type: list[int]
            count: 每个单词在文件中出现的次数，type：list[(string, int)]
            dictionary: 单词的词典，以单词为索引，type: dict[(string, int)]
            rdictionary: 单词的词典，以位置为索引，type: dict[(int, string)]
        """
        self.count = collections.Counter(words).most_common()
        lend, self.index, self.dictionary = 0, list(), dict()
        for word, _ in self.count:
            if word not in self.dictionary:
                self.dictionary[word] = lend
                lend += 1
        for word in words:
            inx = self.dictionary.get(word, -1) # 为所有在文件中出现过的词在词典中查找其的位置并添加索引
            self.index.append(inx)
        self.rdictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def generate_batch(self):
        """
        批量生成数据

        Args:
            None
        Returns:
            words: 返回一个包含多个窗口的语句，shape = [batch_size, window_size]
            target_word: 返回一个跟 words 相对应的目标词，shape = [batch_size]
        """
        words = np.ndarray([self.batch_size, self.window_size], int)
        target_word = np.ndarray([self.batch_size], int)
        for i in range(self.batch_size):
            if self.cur + self.window_size >= self.vocabulary_size:
                self.cur = 0
            words[i] = self.index[self.cur: self.cur + self.window_size]
            target_word[i] = self.index[self.cur + self.window_size]
            self.cur += 1
        return words, target_word
        

if __name__ == "__main__":
    nplm = NPLM("text8", 10, 50, 20, 0.01, 100, 50)
    nplm.test()