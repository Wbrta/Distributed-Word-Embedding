import math
import argparse
import collections
import numpy as np
import tensorflow as tf

class RNNLM(object):
    def __init__(self, filename, hidden_size, learning_rate, step, save_file):
        self.dictionary = None
        with open(filename, 'r') as file:
            words = file.read().split()
        self.text_size = len(words)
        self.build_dataset(words)
        self.step = step
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.save_file = save_file
        self.vocabulary_size = len(self.dictionary)
        self.C = None
        self.W = None
        self.U = None
        self.V = None
        self.b1 = None
        self.b2 = None
        self.word_embedding = None

    def train(self):
        with tf.name_scope('initialize'):
            self.C = tf.Variable(tf.random_uniform([self.vocabulary_size, self.word_embedding_size], -1.0, 1.0))
            self.W = tf.Variable(tf.random_normal(shape = [self.word_embedding_size, self.hidden_size], mean = 0.0, stddev = 0.1))
            self.U = tf.Variable(tf.random_normal(shape = [self.hidden_size, self.hidden_size], mean = 0.0, stddev = 0.1))
            self.V = tf.Variable(tf.random_normal(shape = [self.hidden_size, self.vocabulary_size], mean = 0.0, stddev = 0.1))
            self.b1 = tf.Variable(tf.constant(0.1, shape = [1, self.hidden_size]))
            s = tf.Variable(tf.random_uniform([1, self.hidden_size], -1.0, 1.0))
        for step in range(self.step):
            for layer in range(self.text_size):
                e = tf.nn.embedding_lookup(self.C, layer)
                s = tf.add(tf.add(tf.matmul(e, self.W), tf.matmul(s, self.U)), self.b1)
                y = tf.nn.softmax(tf.add(tf.matmul(s, self.V), self.b2))
                
            

    def build_dataset(self, words):
        self.dictionary, index = dict(), 0
        for word in words:
            if word not in self.dictionary:
                self.dictionary[word] = index
                indedx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()