# encoding:utf-8

import os
import codecs
import collections
from six.moves import cPickle

import numpy as np
import pandas as pd
import re
import itertools
'''
经常遇到在Python程序运行中得到了一些字符串、列表、字典等数据，
想要长久的保存下来，方便以后使用，而不是简单的放入内存中关机断电就丢失数据。
'''

class TextLoader():
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

        train_file = os.path.join(data_dir, "train_set.csv")
        # test_file = os.path.join(data_dir, "test_set.csv")

        self.preprocess(train_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, train_file):

        f = open(train_file)
        reader = pd.read_csv(f, sep=',', iterator=True)
        loop = True
        chunkSize = 10000
        chunks = []
        while loop:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        df_train = pd.concat(chunks, ignore_index=True)


        self.train_x = df_train['word_seg']
        self.train_y = df_train['class']
        # self.test = df_test
    # 构造语言对，前t-1个词作为输入，t作为label
    def create_batches(self):
        self.num_batches = int(len(self.train_x) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        xdata = np.array(self.train_x[:self.num_batches * self.batch_size])
        ydata = np.array(self.train_y[:self.num_batches * self.batch_size])
        # 直接分成（1464×128）
        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

class Loader_test():
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

        # train_file = os.path.join(data_dir, "train_set.csv")
        test_file = os.path.join(data_dir, "test_set.csv")

        self.preprocess(test_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, test_file):

        f = open(test_file)
        reader = pd.read_csv(f, sep=',', iterator=True)
        loop = True
        chunkSize = 10000
        chunks = []
        while loop:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        self.df_test = pd.concat(chunks, ignore_index=True)

        self.test_x = self.df_test['word_seg']
        # self.train_y = df_train['class']
        # self.test = df_test
    # 构造语言对，前t-1个词作为输入，t作为label
    def create_batches(self):
        self.num_batches = int(len(self.test_x) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        xdata = np.array(self.test_x[:self.num_batches * self.batch_size])
        # ydata = np.array(self.train_y[:self.num_batches * self.batch_size])
        # 直接分成（1464×128）
        self.x_batches = np.split(xdata, self.num_batches, 0)
        # self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x = self.x_batches[self.pointer]
        self.pointer += 1
        return x

    def reset_batch_pointer(self):
        self.pointer = 0
