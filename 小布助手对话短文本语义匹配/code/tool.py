import torch
import random
import numpy as np
import csv
import os
import time
import copy
import torch.nn.functional as F
from time import strftime, localtime
import pandas as pd
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# from gensim.models import Word2Vec
from transformers import AdamW, get_linear_schedule_with_warmup
# from tqdm import tqdm
# import pickle
from collections import Counter
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score
from transformers import  RobertaTokenizer



class Vocab(object):
    PAD = 0
    UNK = 1

    def __init__(self, config):
        self.config = config
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None
        self.data_dir = [os.path.join(os.path.dirname(config.data_dir), i) for i in
                         os.listdir(os.path.dirname(config.data_dir))]

    def add_words(self):
        """Add a new token to the vocab and do mapping between word and index.

        Args:
            words (list): The list of tokens to be added.
        """
        words = self.get_all_exampes_words()
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item] if item <= self.size() - 1 else 'UNK'
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return len(self.index2word)

    def build_bert_vocab(self):
        with open(file='vocab', mode="w", encoding="utf-8") as f:
            f.write('[PAD]' + '\n')
            f.write('[CLS]' + '\n')
            f.write('[SEP]' + '\n')
            f.write('[MASK]' + '\n')
            for word, count in self.word2count.most_common():
                f.write(word + '\n')
            f.write('[UNK]')

    def get_all_exampes_words(self):
        words = []
        for dir in self.data_dir:
            with open(file=dir, mode="r", encoding="utf-8") as files:
                for line in files:
                    line = line.strip().split('\t')
                    text_a, text_b = line[0].strip().split(" "), line[1].strip().split(" ")
                    words += text_a
                    words += text_b
        return words

    def get_pre_trained_examples(self):
        examples = []
        for dir in self.data_dir:
            with open(file=dir, mode="r", encoding="utf-8") as files:
                for line in files:
                    line = line.strip().split('\t')
                    text_a, text_b = line[0].strip().split(" "), line[1].strip().split(" ")
                    examples.append(text_a + text_b)
        #                     examples.append(text_b + text_a)

        return examples

    def get_train_dev_test(self):
        train_ls = []
        test_data = []
        for dir in self.data_dir:
            with open(file=dir, mode="r", encoding="utf-8") as files:
                for line in files:
                    line = line.strip().split('\t')
                    if 'train' in dir:
                        text_a, text_b, labal = line[0].strip(), \
                                                line[1].strip(), line[2].strip()
                        train_ls.append((text_a, text_b, labal))
                    else:
                        text_a, text_b = line[0].strip(), \
                                         line[1].strip()
                        test_data.append((text_a, text_b, -100))

        train_data = [d for i, d in enumerate(train_ls) if i % 10 != 0]
        valid_data = [d for i, d in enumerate(train_ls) if i % 10 == 0]
        train_1 = [(x[0],x[1],x[2]) for x in train_data]
        train_2 = [(x[1],x[0],x[2]) for x in train_data]
        valid = [(x[0],x[1],x[2]) for x in valid_data]
        test = [(x[0],x[1],x[2]) for x in test_data]
        if self.config.data_enhance:
            train = train_1 + train_2
        else:
            train = train_1

        return train, valid, test

class BuildDataSet(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        self.tokenizer=BertTokenizer.from_pretrained('vocab')
        self._len = len(self.dataset)

    def __getitem__(self, index):

        text1,text2,label,dense=self.dataset[index]
        text = " ".join([text1,text2]).strip()
        inputs = self.tokenizer(text)
        return (inputs,label,dense)

    def __len__(self):
        return self._len

