# -*- coding: utf8 -*-

import math
import argparse
import collections
import numpy as np
import tensorflow as tf

class NPLM(object):
    """
    神经网络概率语言模型
    """
    def __init__(self, filename, window_size, hidden_size, word_embedding_size, learning_rate, step, batch_size, save_file):
        self.cur = 0
        self.step = step
        self.save_file = save_file
        self.batch_size = batch_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.word_embedding_size = word_embedding_size

        with open(filename, "r") as file:
            words = file.read().split()
        self.words_number = len(words)
        self.build_dataset(words)
        del words
        self.vocabulary_size = len(self.dictionary)

    def sum_target(self, y, index):
        """
        计算 y 的 （i, index[i]) 列之和

        Args:
            y: 一个 m * n 的集合
            index: 列的集合
        Return:
            res[0]: y 的 (i, index[i]) 列之和
        """
        res = tf.zeros([1])
        for inx in index:
            res = tf.add(res, y[inx])
        return res[0]

    def train(self):
        """
        定义训练模型

        Return:
            train: 生成的梯度下降训练模型
            penalized_log_likelihood: 惩罚对数似然，损失函数
        """
        words = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.window_size])
        target_word = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
        with tf.name_scope('input_layer'):
            self.C = tf.Variable(tf.random_uniform([self.vocabulary_size, self.word_embedding_size], -1.0, 1.0))
            index = tf.reshape(words, shape = [self.window_size * self.batch_size])
            e = tf.nn.embedding_lookup(self.C, index)        # shape = [window_size * batch_size, word_embedding_size]
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
            res = tf.zeros([self.batch_size], dtype = tf.float32)
            for i in range(self.batch_size):
                res = tf.add(res, tf.one_hot(i, self.batch_size, on_value = y_[i][target_word[i]]))
        with tf.name_scope('loss_function'):
            regulation = tf.nn.l2_loss(self.H) + tf.nn.l2_loss(self.U)
            penalized_log_likelihood = -(tf.reduce_mean(tf.log(res)) + 1e-5 * regulation)
            train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(penalized_log_likelihood)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(self.step):
                x, y = self.generate_batch()
                sess.run(train, feed_dict = {words: x, target_word: y})
                if step % 100 == 99:
                  print ("Step #%d, penalized_log_likelihood: %f" % ((step + 1), sess.run(penalized_log_likelihood, feed_dict = {words: x, target_word: y})))
            self.word_embedding = sess.run(self.C)

    def save(self):
        """
        保存生成的词向量
        """
        with open(self.save_file, 'w', encoding = 'utf8') as f:
            for word in self.dictionary:
                f.write(word + ": " + str([x for x in self.word_embedding[self.dictionary[word]]]) + "\n")
                f.flush()

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
        words = [[] for i in range(self.batch_size)]
        target_word = [0 for i in range(self.batch_size)]
        for i in range(self.batch_size):
            if self.cur + self.window_size >= self.words_number:
                self.cur = 0
            words[i] = self.index[self.cur: self.cur + self.window_size]
            target_word[i] = self.index[self.cur + self.window_size]
            self.cur += 1
        return np.array(words), np.array(target_word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--corpus', type=str, help="语料文件路径", required=True)
    parser.add_argument('-WS', '--window-size', type=int, help="目标词所需的上文词数", default=10)
    parser.add_argument('-HS', '--hidden-size', type=int, help="神经网络中隐藏层的大小", default=50)
    parser.add_argument('-WES', '--word-embedding-size', type=int, help="词向量的大小", default=20)
    parser.add_argument('-LR', '--learning-rate', type=float, help="学习速率", default=0.01)
    parser.add_argument('-S', '--step', type=int, help="迭代次数", default=10000)
    parser.add_argument('-BS', '--batch-size', type=int, help="批量生成数据的大小", default=50)
    parser.add_argument('-SF', '--save-file', type=str, help="生成的词向量保存的位置", default="data/nplm.txt")
    args = parser.parse_args()

    nplm = NPLM(args.corpus, args.window_size, args.hidden_size, args.word_embedding_size, args.learning_rate, args.step, args.batch_size, args.save_file)
    nplm.train()
    nplm.save()
