# -*- coding: utf8 -*-

import math
import random
import argparse
import collections
import numpy as np
import tensorflow as tf

class SkipGram(object):
    def __init__(self, corpus, vocabulary_size, batch_size, window_size, word_embedding_size, learning_rate, epoches, save_file, num_skips, num_sampled):
        self.cur = 0
        self.epoches = epoches
        self.num_skips = num_skips
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.vocabulary_size = vocabulary_size
        self.word_embedding_size = word_embedding_size
        with open(corpus, 'r') as f:
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
        batch = np.ndarray(shape = (self.batch_size), dtype = int)
        labels = np.ndarray(shape = (self.batch_size, 1), dtype = int)
        buffer = collections.deque(maxlen = self.window_size)
        if self.cur + self.window_size > self.word_number:
            self.cur = 0
        buffer.extend(self.index[self.cur: self.cur + self.window_size])
        self.cur += self.window_size
        for i in range(self.batch_size // self.num_skips):
            context_words = [w for w in range(self.window_size) if w != self.window_size // 2]
            words_to_use = random.sample(context_words, self.num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * self.num_skips + j] = buffer[self.window_size // 2]
                labels[i * self.num_skips + j, 0] = buffer[context_word]
            if self.cur == self.word_number:
                buffer.extend(self.index[0: self.window_size])
                self.cur = self.window_size
            else:
                buffer.append(self.index[self.cur])
                self.cur += 1
        self.cur = (self.cur + self.word_number - self.window_size) % self.word_number
        return batch, labels
 
    def train(self):
        with tf.name_scope('input'):
            inputs = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
            labels = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, 1])
        with tf.name_scope('embedding'):
            C = tf.Variable(tf.random_normal([self.vocabulary_size, self.word_embedding_size], -1.0, 1.0))
            e = tf.nn.embedding_lookup(C, inputs)
        with tf.name_scope('weights'):
            nec_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.word_embedding_size], stddev = 1.0 / math.sqrt(self.word_embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.nce_loss(weights = nec_weights, biases = nce_biases, labels = labels, inputs = e, num_sampled = self.num_sampled, num_classes = self.vocabulary_size))
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoches):
                x, y = self.generate_batch()
                sess.run(optimizer, feed_dict = {inputs: x, labels: y})
                if epoch % 1000 == 999:
                    print ("Epoch #%d, loss: %f" % ((epoch + 1), sess.run(loss, feed_dict = {inputs: x, labels: y})))
            self.word_embedding = sess.run(C)

    def save(self):
        with open(self.save_file, 'w') as f:
            f.write(str(self.vocabulary_size) + " " + str(self.word_embedding_size) + "\n")
            for word in self.dictionary:
                f.write(word + " " + " ".join(map(lambda x: str(x), self.word_embedding[self.dictionary[word]])) + "\n")
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
    parser.add_argument('-NS', '--num-skips', type = int, help = "利用输入多少次以生成 label", default = 2)
    parser.add_argument('-NSD', '--num-sampled', type = int, help = "负采样的数目", default = 64)
    args = parser.parse_args()
 
    sg = SkipGram(args.corpus, args.vocabulary_size, args.batch_size, args.window_size, args.word_embedding_size, args.learning_rate, args.epoches, args.save_file, args.num_skips, args.num_sampled)
    sg.train()
    sg.save()
