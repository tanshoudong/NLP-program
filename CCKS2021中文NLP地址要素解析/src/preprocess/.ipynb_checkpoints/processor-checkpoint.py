import os
import re
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.pseudo = pseudo
        self.distant_labels = distant_labels


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class CRFFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        super(CRFFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        # labels
        self.labels = labels

        # pseudo
        self.pseudo = pseudo

        # distant labels
        self.distant_labels = distant_labels


class SpanFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 start_ids=None,
                 end_ids=None,
                 pseudo=None):
        super(SpanFeature, self).__init__(token_ids=token_ids,
                                          attention_masks=attention_masks,
                                          token_type_ids=token_type_ids)
        self.start_ids = start_ids
        self.end_ids = end_ids
        # pseudo
        self.pseudo = pseudo

class MRCFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 ent_type=None,
                 start_ids=None,
                 end_ids=None,
                 pseudo=None):
        super(MRCFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        self.ent_type = ent_type
        self.start_ids = start_ids
        self.end_ids = end_ids

        # pseudo
        self.pseudo = pseudo


class NERProcessor:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.online_dir = './tcdata'
        self.pseudo_dir = './data/pseudo_data/pseudo.txt'
        self.online_files_names = ['train.conll','dev.conll','final_test.txt']

    def get_data_examples(self):
        train_dir,dev_dir,test_dir = [os.path.join(self.online_dir,x) for x in self.online_files_names]
        train_feature,dev_feature,fu_test_feature,pseudo_feature,chu_test_feature = [],[],[],[],[]

        with open(file=train_dir,mode='r',encoding='utf-8') as files:
            tmp_char,tmp_ids = [],[]
            for line in files:
                if line.strip():
                    tmp_line = line.strip().split(' ')
                    tmp_char.append(tmp_line[0].strip())
                    tmp_ids.append(tmp_line[1].strip())
                else:
                    assert len(tmp_char)==len(tmp_ids)
                    train_feature.append(Feature(tmp_char,tmp_ids))
                    tmp_char, tmp_ids = [], []
            if len(tmp_char)>0 and len(tmp_char)==len(tmp_ids):
                train_feature.append(Feature(tmp_char, tmp_ids))

        with open(file=dev_dir,mode='r',encoding='utf-8') as files:
            tmp_char,tmp_ids = [],[]
            for line in files:
                if line.strip():
                    tmp_line = line.strip().split(' ')
                    tmp_char.append(tmp_line[0].strip())
                    tmp_ids.append(tmp_line[1].strip())
                else:
                    assert len(tmp_char)==len(tmp_ids)
                    dev_feature.append(Feature(tmp_char,tmp_ids))
                    tmp_char, tmp_ids = [], []
            if len(tmp_char)>0 and len(tmp_char)==len(tmp_ids):
                dev_feature.append(Feature(tmp_char, tmp_ids))

        with open(file=test_dir, mode='r', encoding='utf-8') as files:
            for line in files:
                if not line.strip():
                    continue
                text = line.strip().split('\u0001')[-1].strip()
                fu_test_feature.append(Feature(list(text),None))

        with open(file=self.pseudo_dir, mode='r', encoding='utf-8') as files:
            for line in files:
                if line.strip():
                    _,text,label = line.strip().split('\u0001')
                    pseudo_feature.append(Feature(list(text),label.strip().split(' '),True))


        with open(file='./data/raw_data/final_test.txt', mode='r', encoding='utf-8') as files:
            for line in files:
                if not line.strip():
                    continue
                text = line.strip().split('\u0001')[-1].strip()
                chu_test_feature.append(Feature(list(text),None))


        return train_feature,dev_feature,fu_test_feature,pseudo_feature,chu_test_feature

    @staticmethod
    def _refactor_labels(sent, labels, distant_labels, start_index):
        """
        分句后需要重构 labels 的 offset
        :param sent: 切分并重新合并后的句子
        :param labels: 原始文档级的 labels
        :param distant_labels: 远程监督 label
        :param start_index: 该句子在文档中的起始 offset
        :return (type, entity, offset)
        """
        new_labels, new_distant_labels = [], []
        end_index = start_index + len(sent)

        for _label in labels:
            if start_index <= _label[2] <= _label[3] <= end_index:
                new_offset = _label[2] - start_index

                assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

                new_labels.append((_label[1], _label[-1], new_offset))
            # label 被截断的情况
            elif _label[2] < end_index < _label[3]:
                raise RuntimeError(f'{sent}, {_label}')

        for _label in distant_labels:
            if _label in sent:
                new_distant_labels.append(_label)

        return new_labels, new_distant_labels

    def get_examples(self, raw_examples, set_type):
        examples = []

        for i, item in enumerate(raw_examples):
            text = item['text']
            distant_labels = item['candidate_entities']
            pseudo = item['pseudo']

            sentences = cut_sent(text, self.cut_sent_len)
            start_index = 0

            for sent in sentences:
                labels, tmp_distant_labels = self._refactor_labels(sent, item['labels'], distant_labels, start_index)

                start_index += len(sent)

                examples.append(InputExample(set_type=set_type,
                                             text=sent,
                                             labels=labels,
                                             pseudo=pseudo,
                                             distant_labels=tmp_distant_labels))

        return examples

class Feature:
    def __init__(self,text,label,pseudo=None):
        self.text = text
        self.label = label
        self.pseudo = pseudo


if __name__ == '__main__':
    pass
