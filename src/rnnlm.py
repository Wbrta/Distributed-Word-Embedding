import math
import argparse
import collections
import numpy as np
import tensorflow as tf

class RNNLM(object):
    def __init__(self, corpus, window_size, hidden_size, word_embedding_size, learning_rate, step, save_file, threshold):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.word_embedding_size = word_embedding_size
        self.learning_rate = learning_rate
        self.step = step
        self.save_file = save_file
        self.threshold = threshold
        with open(corpus, 'r') as file:
            self.words = file.read().split()
        self.build_dataset(self.words)
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

    def train(self):
        with tf.name_scope('initialize'):
            self.C = tf.Variable(tf.random_uniform([self.vocabulary_size, self.word_embedding_size], -1.0, 1.0))
            self.W = tf.Variable(tf.random_normal(shape = [self.word_embedding_size, self.hidden_size], mean = 0.0, stddev = 0.1))
            self.U = tf.Variable(tf.random_normal(shape = [self.hidden_size, self.hidden_size], mean = 0.0, stddev = 0.1))
            self.V = tf.Variable(tf.random_normal(shape = [self.hidden_size, self.vocabulary_size], mean = 0.0, stddev = 0.1))
            self.b1 = tf.Variable(tf.constant(0.1, shape = [1, self.hidden_size]))
            s = tf.Variable(tf.random_uniform([1, self.hidden_size], -1.0, 1.0))
        for step in range(self.step):
            for layer in range():
                e = tf.nn.embedding_lookup(self.C, layer)
                s = tf.add(tf.add(tf.matmul(e, self.W), tf.matmul(s, self.U)), self.b1)
                y = tf.nn.softmax(tf.add(tf.matmul(s, self.V), self.b2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--corpus', type=str, help="语料文件路径", required=True)
    parser.add_argument('-WS', '--window-size', type=int, help="目标词所需的上文词数", default=10)
    parser.add_argument('-HS', '--hidden-size', type=int, help="神经网络中隐藏层的大小", default=50)
    parser.add_argument('-WES', '--word-embedding-size', type=int, help="词向量的大小", default=20)
    parser.add_argument('-LR', '--learning-rate', type=float, help="学习速率", default=0.01)
    parser.add_argument('-S', '--step', type=int, help="迭代次数", default=10000)
    parser.add_argument('-SF', '--save-file', type=str, help="生成的词向量保存的位置", default="nplm.txt")
    parser.add_argument('-T', '--threshold', type=int, help="阈值，语料中的词出现次数少于此值的均设为<UNK>", default=5)
    args = parser.parse_args()

    rnnlm = RNNLM()
    rnnlm.train()