import argparse
import collections
import numpy as np
import tensorflow as tf

class RNNLM(object):
    def __init__(self):
        # 字典相关
        self.index = None
        self.count = None
        self.dictionary = None
        self.rdictionary = None
        self.vocabulary_size = None
        # 模型相关
        self.hidden_size = None

    def train(self):
        with tf.name_scope('input_layer'):
            self.C = tf.Variable(tf.random_uniform([self.vocabulary_size, self.word_embedding_size], -1.0, 1.0))
            e = tf.nn.embedding_lookup(self.C, )
        with tf.name_scope('hidden_layer'):
            self.H = tf.Variable(tf.truncated_normal(shape = [self.vocabulary_size, self.hidden_size], stddev = 1.0 / math.sqrt(self.word_embedding_size)))
            self.b1 = tf.Variable(tf.constant(0.1, shape = [1, self.hidden_size]))
            s = tf.add(tf.matmul(e, self.H), self.b1)
        with tf.name_scope('output_layer'):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()