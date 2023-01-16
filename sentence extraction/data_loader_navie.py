# -*- coding: utf-8 -*-

"""Data loader"""

import random
import numpy as np
import os
import sys

import torch

from transformers import RobertaTokenizer, BertTokenizer, BartTokenizer
import utils

special_tokens_dict = {'eos_token': '<EOS>'}


class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0
        self.classi = params.classi

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}

        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tag_pad_idx = self.tag2idx['O']

        'NEED CHANGE'
        #self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
        #self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tag')
        with open(file_path, 'r') as file:
            for line in file:
                tags.append(line.strip())
        tags = ['O','M', 'T','MT']
        return tags

    def readFile(self, file):
        sentences = []
        tags = []

        for i, line in enumerate(file):
            # replace each token by its index
            sp = line.strip().split("\t")

            tokens = sp[1].split(" ")
            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')
            sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))

            tag_id = self.tag2idx.get(sp[2])
            tags.append(tag_id)

        return sentences, tags

    def load_sentences_tags(self, sentences_file, d, category='train'):
        """Loads sentences and tags from their corresponding files.
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        with open(sentences_file, 'r', errors="ignore") as file:
            sentences, tags = self.readFile(file)
            assert len(sentences) == len(tags)
            # storing sentences and tags in dict d
            d['data'] = sentences
            d['tags'] = tags
            d['size'] = len(sentences)

    def load_data(self, data_type, category=None, transback_file=None, random_file=None, wordEmb_file=None,
                            tfidf_file=None, contextEmb_file=None):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}

        # if data_type in ['train', 'val', 'test']:
        sentences_file = os.path.join(self.data_dir, data_type)

        self.load_sentences_tags(sentences_file, data, category=category)
        print("load_data is OK: " + data_type)
        # else:
        #    raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data, shuffle=False, augClassi=None):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.shuffle(order)

        # one pass over data
        for i in range(0, (data['size'] + self.batch_size - 1) // self.batch_size):
            # fetch sentences and tags

            if (i + 1) * self.batch_size < data['size']:
                sentences = [data['data'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
                tags = [data['tags'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
            else:
                if data['size']-i * self.batch_size<=3:
                    sentences = [data['data'][idx] for idx in order[i * self.batch_size:]]
                    tags = [data['tags'][idx] for idx in order[i * self.batch_size:]]
                    sentences.extend([data['data'][idx] for idx in order[i * self.batch_size:]])
                    tags.extend([data['tags'][idx] for idx in order[i * self.batch_size:]])
                    sentences.extend([data['data'][idx] for idx in order[i * self.batch_size:]])
                    tags.extend([data['tags'][idx] for idx in order[i * self.batch_size:]])
                    sentences.extend([data['data'][idx] for idx in order[i * self.batch_size:]])
                    tags.extend([data['tags'][idx] for idx in order[i * self.batch_size:]])
                else:
                    sentences = [data['data'][idx] for idx in order[i * self.batch_size:]]
                    tags = [data['tags'][idx] for idx in order[i * self.batch_size:]]

            # # batch lengths
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)  # should change

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_tags = np.zeros((batch_len))

            # copy the data to the numpy array
            batch_lengths = []
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                    batch_tags[j] = tags[j]
                    batch_lengths.append(cur_len)
                else:
                    batch_data[j] = sentences[j][:max_len]
                    batch_tags[j] = tags[j]
                    batch_lengths.append(max_len)

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            batch_lengths = torch.tensor(batch_lengths, dtype=torch.long)
            # shift tensors to GPU if available

            batch_data, batch_tags, batch_lengths = \
                    batch_data.to(self.device), batch_tags.to(self.device), batch_lengths.to(self.device)

            yield batch_data, batch_tags, max_len, batch_lengths

