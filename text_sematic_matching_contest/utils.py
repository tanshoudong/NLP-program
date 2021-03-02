import torch
import random
import numpy as np
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def convert_examples_to_features(examples,label_list,pad_token=0,pad_token_segment_id=0):
    """
    :param examples: List [ sentences1,sentences2,label, category]
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :param label_list: List of labels.
    :param max_length: Maximum example length
    :param pad_token: 0
    :param pad_token_segment_id: 0
    :return: [(example.guid, input_ids, attention_mask, token_type_ids, label), ......]
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example[0], example[1], add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        if example[2] is not None:
            label = label_map[example[2]]
        else:
            label = 0
        features.append(
            InputFeatures(input_ids, attention_mask, token_type_ids, label)
        )

    return features



class DataProcessor:
    def __init__(self, config):
        self.data_dir = config.data_dir

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_train_examples(self):
        return self._read_csv(self.data_dir)

    def _read_csv(self, input_file):
        data_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            tsv_list = list(csv.reader(f, delimiter='\t'))
            for line in tsv_list:
                text_a,text_b,label=line
                data_list.append([text_a, text_b, label])
        return data_list

    def train_dev_split(self,data):
        df=pd.DataFrame(data,columns=['text_left','text_right',"label"])
        train,valid=train_test_split(df,test_size=0.05, stratify=df['label'])
        return train,valid


class train_vector:
    def __init__(self,config):
        self.config=config
        self.data_dir=[os.path.join(os.path.dirname(self.config.data_dir),i) for i in
        os.listdir(os.path.dirname(self.config.data_dir))]
        self.w2v=self.train_w2v()

    def train_w2v(self,L=128):
        tmp_dir=os.path.join(self.config.embed_dir, 'w2v' + ".{}d".format(L))
        if os.path.isfile(tmp_dir):
            return pickle.load(open(tmp_dir,'rb'))
        sentences = []
        for dir in self.data_dir:
            with open(file=dir,mode="r",encoding="utf-8") as files:
                for line in files:
                    line=line.strip().split('\t')
                    text_a,text_b=line[0].strip().split(" "),line[1].strip().split(" ")
                    sentences.append(text_a)
                    sentences.append(text_b)
        print("Sentence Num {}".format(len(sentences)))
        w2v = Word2Vec(sentences, size=L, window=4, min_count=1, sg=1, workers=32, iter=10)
        print("save w2v to {}".format(tmp_dir))
        pickle.dump(w2v, open(tmp_dir,'wb'))
        return w2v


