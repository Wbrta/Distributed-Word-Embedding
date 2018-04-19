# -*- coding: utf8 -*-

import argparse
import collections
import numpy as np
import tensorflow as tf

class nplm(object):
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

    def __init__(self, filename, window_size, hidden_size, word_embedding_size, learning_rate):
        with open(filename, "r") as file:
            text = file.read()
        words = text.split()
        build_dataset(words)
        del text, words
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.vocabulary_size = len(self.dictionary)
        self.word_embedding_size = word_embedding_size
    
    def train(self):
        with tf.name_scope('input_layer'):
            X = tf.placeholder(tf.float32, shape = (1, self.window_size * self.word_embedding_size))
            H = tf.truncated_normal(shape = (self.window_size * self.word_embedding_size, self.hidden_size))
            b1 = tf.Variable(tf.constant(0.1, shape = (1, self.hidden_size)))
        with tf.name_scope('hidden_layer'):
            h = tf.tanh(tf.add(tf.matmul(X, H), b1))
            U = tf.truncated_normal(shape = (self.hidden_size, self.vocabulary_size))
            b2 = tf.Variable(tf.constant(0.1, shape = (1, self.vocabulary_size)))
        with tf.name_scope('output_layer'):
            y = tf.nn.softmax(tf.add(tf.matmul(h, U), b2))
        with tf.name_scope('loss_function'):
            loss = -y[0][target_word]
            train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        with tf.name_scope('perplexity'):
            regulation = tf.nn.l2_loss(H) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(U) + tf.nn.l2_loss(b2)
            perplexity = -(tf.reduce_mean(tf.log()) + 1e-5 * regulation)
        return train, perplexity

    def test(self):
        pass

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
        for word, _ in count:
            if word not in dictionary:
                dictionary[word] = lend
                lend += 1
        for word in words:
            inx = dictionary.get(word, -1) # 为所有在文件中出现过的词在词典中查找其的位置并添加索引
            index.append(inx)
        self.rdictionary = dict(zip(dictionary.values(), dictionary.keys()))

    def generate_batch(self):
        pass
        

if __name__ == "__main__":
    pass