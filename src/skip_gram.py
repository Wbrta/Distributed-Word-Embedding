# -*- coding: utf8 -*-

import math
import argparse
import collections
import numpy as np
import tensorflow as tf

class SkipGram(object):
    def __init__(self, corpus, window_size, word_embedding_size, learning_rate, step, batch_size, save_file):
        pass
    def train(self):
        pass
    def save(self):
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--corpus', type=str, help="语料文件路径", required=True)
    parser.add_argument('-WS', '--window-size', type=int, help="目标词所需的上文词数", default=10)
    parser.add_argument('-WES', '--word-embedding-size', type=int, help="词向量的大小", default=20)
    parser.add_argument('-LR', '--learning-rate', type=float, help="学习速率", default=0.01)
    parser.add_argument('-S', '--step', type=int, help="迭代次数", default=10000)
    parser.add_argument('-BS', '--batch-size', type=int, help="批量生成数据的大小", default=50)
    parser.add_argument('-SF', '--save-file', type=str, help="生成的词向量保存的位置", default="data/skip_gram.txt")
    args = parser.parse_args()

    sg = SkipGram(args.corpus, args.window_size, args.word_embedding_size, args.learning_rate, args.step, args.batch_size, args.save_file)
    sg.train()
    sg.save()