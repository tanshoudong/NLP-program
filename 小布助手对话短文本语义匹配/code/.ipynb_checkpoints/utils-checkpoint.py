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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# from gensim.models import Word2Vec
from transformers import AdamW, get_linear_schedule_with_warmup
# from tqdm import tqdm
# import pickle
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from collections import Counter
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score
from transformers import  RobertaTokenizer
import itertools


def compute_loss(outputs, labels, loss_method='binary'):
    loss = 0.
    if loss_method == 'binary':
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels)
    elif loss_method == 'cross_entropy':
        loss = F.cross_entropy(outputs, labels)
    else:
        raise Exception("loss_method {binary or cross_entropy} error. ")
    return loss


def train_dev_test_for_mlm(train,dev,test):
    df = pd.concat([dev,test])
    df["label"] = -1
    tmp = pd.DataFrame()
    tmp["text_left"], tmp["text_right"], tmp["label"] = df["text_right"], df["text_left"], df["label"]
    all = pd.concat([train,df,tmp])
    return all


def set_seed(args=None):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    cudnn.deterministic = True
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(12345)



class DataProcessor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config=config

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
        if self.config.data_enhance:
            tmp = pd.DataFrame()
            tmp["text_left"], tmp["text_right"], tmp["label"] = train["text_right"], train["text_left"], train["label"]
            train = pd.concat([train,tmp])
        return train,valid

    def get_test_data(self,test_dir):
        test = pd.read_csv(test_dir,sep='\t',names=['text_left','text_right'])
        test["label"] = -2
        return test



class train_vector:
    def __init__(self,config):
        self.config=config
        self.data_dir=[os.path.join(os.path.dirname(self.config.data_dir),i) for i in
        os.listdir(os.path.dirname(self.config.data_dir))]
        self.w2v=self.train_w2v()
        self.vocab =  None


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
                    sentences.append(text_a + text_b)
                    sentences.append(text_b + text_a)

        print("Sentence Num {}".format(len(sentences)))
        w2v = Word2Vec(sentences, size=L, window=4, min_count=1, sg=1, workers=32, iter=10)
        print("save w2v to {}".format(tmp_dir))
        pickle.dump(w2v, open(tmp_dir,'wb'))
        return w2v


    def collate_fn_v1(self,batch_data):
        batch_size = len(batch_data)
        max_len = max([len(x[0]) for x in batch_data])
        # for x, y in batch_data:
        #     input_ids.append(x + (max_len - len(x)) * [0])
        #     label.append(int(y))
        inputs = np.zeros(shape=(batch_size,max_len,self.w2v.wv.vector_size))
        mask = np.zeros(shape=(batch_size,max_len))
        output_id = np.zeros(shape=(batch_size,max_len))-100

        label=[]
        # 选择20%的token进行掩码，其中80%设为[mask], 10%设为自己,10%随机选择
        for i,(x,y) in enumerate(batch_data):
            for j,word in enumerate(x):
                mask[i, j] = 1
                if random.random() < 0.2:
                    prob = random.random()
                    if prob < 0.8:
                        output_id[i, j] = self.vocab[word]
                    elif prob < 0.9:
                        inputs[i, j] = self.w2v.wv[word]
                        output_id[i, j] = self.vocab[word]
                    else:
                        random_word = random.choice(list(self.w2v.wv.vocab.keys()))
                        inputs[i, j] = self.w2v.wv[random_word]
                        output_id[i, j] = self.vocab[random_word]
                else:
                    inputs[i,j] = self.w2v.wv[word]
            label.append(int(y))

        inputs = torch.tensor(data=inputs).type(torch.FloatTensor)
        mask = torch.tensor(data=mask).type(torch.LongTensor)
        output_id = torch.tensor(data=output_id).type(torch.LongTensor)
        label = torch.tensor(data=label).type(torch.FloatTensor)
        return inputs,mask,output_id,label


    def get_all_exampes_words(self):
        words=[]
        for dir in self.data_dir:
            with open(file=dir,mode="r",encoding="utf-8") as files:
                for line in files:
                    line=line.strip().split('\t')
                    text_a,text_b=line[0].strip().split(" "),line[1].strip().split(" ")
                    words+=text_a
                    words+=text_b
        return words



class Vocab(object):
    PAD = 0
    UNK = 1

    def __init__(self,config):
        self.config = config
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>','<UNK>']
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
            return self.index2word[item] if item <= self.size()-1 else 'UNK'
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return len(self.index2word)

    def build_bert_vocab(self):
        with open(file='../user_data/vocab', mode="w", encoding="utf-8") as f:
            f.write('[PAD]'+'\n')
            f.write('[CLS]'+'\n')
            f.write('[SEP]'+'\n')
            f.write('[MASK]'+'\n')
            for word,count in self.word2count.most_common():
                f.write(word+'\n')
            f.write('[UNK]')

    def get_all_exampes_words(self):
        words=[]
        for dir in self.data_dir:
            with open(file=dir,mode="r",encoding="utf-8") as files:
                for line in files:
                    line=line.strip().split('\t')
                    text_a,text_b=line[0].strip().split(" "),line[1].strip().split(" ")
                    words+=text_a
                    words+=text_b
        return words

    def get_pre_trained_examples(self):
        examples = []
        for dir in self.data_dir:
            with open(file=dir,mode="r",encoding="utf-8") as files:
                for line in files:
                    line=line.strip().split('\t')
                    text_a,text_b=line[0].strip().split(" "),line[1].strip().split(" ")
                    examples.append(text_a + text_b)
        return examples

    def get_train_dev_test(self):
        train_ls = []
        for dir in self.data_dir:
            if 'train' in dir:
                with open(file=dir,mode="r",encoding="utf-8") as files:
                    for line in files:
                        line=line.strip().split('\t')
                        text_a,text_b,labal=line[0].strip(),\
                                                line[1].strip(),line[2].strip()
                        train_ls.append((text_a,text_b,labal))
                
#         train_data = [d for i, d in enumerate(train_ls) if i % 10 != 0]
#         valid_data = [d for i, d in enumerate(train_ls) if i % 10 == 0]

        # if self.config.close_bag_enhance:
        #     train_data = data_aug(train_ls)
        # else:
        #     train_data = train_ls
        
        return train_ls

#         train1 = [(x[0]+' '+x[1],x[2]) for x in train_data]
#         train2 = [(x[1]+' '+x[0],x[2]) for x in train_data]
        
#         train = train1 + train2
        
#         valid = [(x[0]+' '+x[1],x[2]) for x in valid_data]
#         test = [(x[0]+' '+x[1],x[2]) for x in test_data]
        
#         return train,valid,test






class BuildDataSet(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        self.tokenizer=BertTokenizer.from_pretrained('../user_data/vocab')
        self._len = len(self.dataset)

    def __getitem__(self, index):

        example=self.dataset[index]
        example,label = random_mask(example)
        inputs = self.tokenizer(" ".join(example), return_tensors="pt")
        label = self.tokenizer(" ".join(label))["input_ids"]
        return inputs,label

    def __len__(self):
        return self._len




def collate_fn(batch_data,padding_token=0,pad_token_segment_id=0):
    max_len = max([len(x[1]) for x in batch_data])
    input_ids, token_type_ids, attention_mask, label=[],[],[],[]
    for x,y in batch_data:
        x,z,w = x["input_ids"],x["token_type_ids"],x["attention_mask"]
        input_ids.append(list(x.numpy()[0])+(max_len-len(x[0]))*[padding_token])
        token_type_ids.append(list(z.numpy()[0])+(max_len-len(z[0]))*[pad_token_segment_id])
        attention_mask.append(list(w.numpy()[0])+(max_len-len(w[0]))*[0])
        label.append(y+(max_len-len(y))*[0])

    input_ids = torch.tensor(data=input_ids).type(torch.LongTensor)
    token_type_ids = torch.tensor(data=token_type_ids).type(torch.LongTensor)
    attention_mask = torch.tensor(data=attention_mask).type(torch.LongTensor)
    label = torch.tensor(data=label).type(torch.LongTensor)
    return input_ids, token_type_ids, attention_mask, label


def random_mask(text_ids):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.20 * 0.8:
            input_ids.append('[MASK]')
            output_ids.append(i)
        elif r < 0.20 * 0.9:
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.20:
            input_ids.append(str(np.random.choice(20600)))
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append('[PAD]')
    return input_ids, output_ids





def model_evaluate(config,model,data_iter):
    model.eval()
    predict_all = []
    label_all = []

    with torch.no_grad():
        for batch, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_iter):
            input_ids = input_ids.cuda(config.local_rank, non_blocking=True)
            attention_mask = attention_mask.cuda(config.local_rank, non_blocking=True)
            token_type_ids = token_type_ids.cuda(config.local_rank, non_blocking=True)
            label = label.cuda(config.local_rank, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=label)

            loss = outputs.loss
            logits = outputs.logits
            pred_pob = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            predict_all.extend(list(pred_pob.detach().cpu().numpy()))
            label_all.extend(list(label.detach().cpu().numpy()))

        dev_auc = roc_auc_score(label_all, predict_all)
        
        return torch.tensor(dev_auc).cuda(config.local_rank, non_blocking=True)





def submit_result(ls):
    with open(file='result.tsv',mode="w",encoding="utf-8") as f:
        for line in ls:
            f.write(str(line)+'\n')


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
        
        
        
def data_aug(datas):
    dic = {}
    for data in datas:
        if data[0] not in dic:
            dic[data[0]] = {'true': [], 'false': []}
            dic[data[0]]['true' if data[2] == '1' else 'false'].append(data[1])
        else:
            dic[data[0]]['true' if data[2] == '1' else 'false'].append(data[1])


    new_datas = []
    for sent1, sent2s in dic.items():
        trues = sent2s['true']
        falses = sent2s['false']
        # 还原原始数据
        for true in trues:
            new_datas.append((sent1,true,'1'))
        for false in falses:
            new_datas.append((sent1,false,'0'))
        temp_trues = []
        temp_falses = []
        if len(trues) != 0 and len(falses) != 0:
            ori_rate = len(trues) / len(falses)
            # 相似数据两两交互构造新的相似对
            for i in itertools.combinations(trues, 2):
                temp_trues.append((i[0],i[1],'1'))
            # 构造不相似数据
            for true in trues:
                for false in falses:
                    temp_falses.append((true,false,'0'))
            num_t = int(len(temp_falses) * ori_rate)
            num_f = int(len(temp_trues) / ori_rate)
            temp_rate = len(temp_trues) / len(temp_falses)
            # if ori_rate < temp_rate:
            #     temp_trues = temp_trues[: num_t]
            # else:
            #     temp_falses = temp_falses[: num_f]
        new_datas = new_datas + temp_trues + temp_falses
    return new_datas