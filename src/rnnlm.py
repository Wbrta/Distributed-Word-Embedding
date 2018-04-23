import math
import argparse
import collections
import numpy as np
import tensorflow as tf

class RNNLM(object):
    def __init__(self, filename, hidden_size, learning_rate, save_file):
        self.dictionary = None
        with open(filename, 'r') as file:
            words = file.read().split()
        self.build_dataset(words)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.save_file = save_file

    def train(self):
        pass

    def build_dataset(self, words):
        self.dictionary, index = dict(), 0
        for word in words:
            if word not in self.dictionary:
                self.dictionary[word] = index
                indedx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()