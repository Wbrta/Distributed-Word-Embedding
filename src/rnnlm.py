# -*- coding: utf8 -*-

import random
import argparse
import collections
import numpy as np
import tensorflow as tf

class RNNLM(object):
    def __init__(self, corpus, batch_size, hidden_size, word_embedding_size, learning_rate, epoches, save_file, threshold):
        self.cur = 0
        self.batch_size = batch_size
        self.epoches = epoches
        self.save_file = save_file
        self.threshold = threshold
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.word_embedding_size = word_embedding_size
        with open(corpus, 'r') as f:
            words = f.read().split()
        self.build_dataset(words)
        self.word_number = len(words)
        del words
        self.vocabulary_size = len(self.dictionary)

    def build_dataset(self, words):
        self.count = [['UNK', -1]]
        new_tmp, cont = [], 0
        temp = collections.Counter(words).most_common()
        for word, cnt in temp:
            if cnt >= self.threshold:
                new_tmp.append((word, cnt))
            else:
                cont += cnt
        self.count.extend(new_tmp)
        self.count[0][1] = cont
        self.dictionary, self.index, cont = dict(), list(), 1
        self.dictionary['UNK'] = 0
        for word, _ in self.count:
            if word not in self.dictionary:
                self.dictionary[word] = cont
                cont += 1
        for word in words:
            inx = self.dictionary.get(word, 0)
            self.index.append(inx)
        self.rdictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def generate_batch(self):
        def one_hot(index, depth):
            return [0 if i != index else 1 for i in range(depth)]
        if self.cur + self.batch_size + 1 >= self.word_number:
            self.cur = 0
        x = np.array(self.index[self.cur: self.cur + self.batch_size])
        tmp = self.index[self.cur + 1: self.cur + 1 + self.batch_size]
        self.cur += random.randint(1, self.batch_size)
        y = np.array([one_hot(inx, self.vocabulary_size) for inx in tmp])
        return x, y

    def train(self):
        C = tf.Variable(tf.random_uniform([self.vocabulary_size, self.word_embedding_size]))
        W = tf.Variable(tf.random_normal(shape = [self.word_embedding_size, self.hidden_size], mean = 0.0, stddev = 0.1))
        U = tf.Variable(tf.random_normal(shape = [self.hidden_size, self.hidden_size], mean = 0.0, stddev = 0.1))
        V = tf.Variable(tf.random_normal(shape = [self.hidden_size, self.vocabulary_size], mean = 0.0, stddev = 0.1))
        s = tf.Variable(tf.random_uniform([1, self.hidden_size], -1.0, 1.0))

        loss = tf.zeros([1])
        word = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
        target = tf.placeholder(dtype = tf.float32, shape = [self.batch_size, self.vocabulary_size])
        for i in range(self.batch_size):
            e = tf.nn.embedding_lookup(C, word[i])
            e = tf.reshape(e, shape = [1, self.word_embedding_size])
            s = tf.nn.sigmoid(tf.matmul(e, W) + tf.matmul(s, U))
            y_ = tf.matmul(s, V)
            loss = tf.add(loss, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target[i], logits=y_)))
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoches):
                x, y = self.generate_batch()
                sess.run(train, feed_dict = {word: x, target: y})
                print ("Epoch #%d, loss: %f" % ((epoch + 1), sess.run(loss, feed_dict = {word: x, target: y})))
            self.word_embedding = sess.run(C)

    def save(self):
        with open(self.save_file, 'w', encoding = 'utf8') as f:
            for word in self.dictionary:
                f.write(word + ": " + str([x for x in self.word_embedding[self.dictionary[word]]]) + "\n")
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--corpus', type = str, help = "语料文件路径", required = True)
    parser.add_argument('-BS', '--batch-size', type = int, help = "批量生成数据的大小", default = 100)
    parser.add_argument('-HS', '--hidden-size', type = int, help = "神经网络中隐藏层的大小", default = 50)
    parser.add_argument('-WES', '--word-embedding-size', type = int, help = "词向量的大小", default = 20)
    parser.add_argument('-LR', '--learning-rate', type = float, help = "学习速率", default = 0.01)
    parser.add_argument('-E', '--epoches', type = int, help = "迭代次数", default = 10000)
    parser.add_argument('-SF', '--save-file', type = str, help = "生成的词向量保存的位置", default = "rnnlm.txt")
    parser.add_argument('-T', '--threshold', type = int, help = "阈值，语料中的词出现次数少于此值的均设为<UNK>", default = 5)
    args = parser.parse_args()

    rnnlm = RNNLM(args.corpus, args.batch_size, args.hidden_size, args.word_embedding_size, args.learning_rate, args.epoches, args.save_file, args.threshold)
    rnnlm.train()
    rnnlm.save()