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
from gensim.models import Word2Vec
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pickle
from collections import Counter
from transformers import BertTokenizer
import logging
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>','<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None

    def add_words(self, words):
        """Add a new token to the vocab and do mapping between word and index.

        Args:
            words (list): The list of tokens to be added.
        """
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings

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
        with open(file='vocab', mode="w", encoding="utf-8") as f:
            f.write('[PAD]'+'\n')
            f.write('[CLS]'+'\n')
            f.write('[SEP]'+'\n')
            f.write('[MASK]'+'\n')
            for word,count in self.word2count.most_common():
                f.write(word+'\n')
            f.write('[UNK]')


class BuildDataSet(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset.values
        self.tokenizer=BertTokenizer.from_pretrained('vocab')
        self._len = len(self.dataset)

    def __getitem__(self, index):
        example=self.dataset[index]
        text_a=example[0].strip().split(" ")
        text_b=example[1].strip().split(" ")
        label=example[2]
        encode=self.tokenizer.encode_plus(text_a,text_b,add_special_tokens=True)
        input_ids,token_type_ids,attention_mask=encode["input_ids"],encode["token_type_ids"],encode["attention_mask"]
        return input_ids,token_type_ids,attention_mask,label

    def __len__(self):
        return self._len


def collate_fn(batch_data,padding_token=0,pad_token_segment_id=0):
    max_len=max([len(x[0]) for x in batch_data])
    input_ids, token_type_ids, attention_mask, label=[],[],[],[]
    for x,y,z,w in batch_data:
        input_ids.append(x+(max_len-len(x))*[padding_token])
        token_type_ids.append(y+(max_len-len(y))*[pad_token_segment_id])
        attention_mask.append(z+(max_len-len(z))*[0])
        label.append(int(w))

    input_ids = torch.tensor(data=input_ids).type(torch.LongTensor)
    token_type_ids = torch.tensor(data=token_type_ids).type(torch.LongTensor)
    attention_mask = torch.tensor(data=attention_mask).type(torch.LongTensor)
    label = torch.tensor(data=label).type(torch.FloatTensor)
    return input_ids, token_type_ids, attention_mask, label


def train_process(config, model, train_iter, dev_iter=None):
    start_time = time.time()
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    diff_part = ["bert.embeddings", "bert.encoder"]

    if config.diff_learning_rate is False:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    else:
        logger.info("use the diff learning rate")
        # the formal is basic_bert part, not include the pooler
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.head_learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.head_learning_rate
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    t_total = len(train_iter) * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", config.train_num_examples)
    logger.info("  Dev Num examples = %d", config.dev_num_examples)
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s", config.device)

    global_batch = 0  # 记录进行到多少batch
    dev_best_auc = 0
    last_batch_improve = 0  # 记录上次验证集loss下降的batch数
    last_epoch_improve = 0
    flag = False  # 记录是否很久没有效果提升

    predict_all = []
    labels_all = []
    best_model = copy.deepcopy(model)

    for epoch in range(config.num_train_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))

        for batch, (input_ids, token_type_ids, attention_mask, label) in enumerate(train_iter):
            global_batch += 1
            model.train()
            input_ids = input_ids.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = label.to(config.device)
            outputs,loss = model(input_ids,token_type_ids,attention_mask,labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            outputs = outputs.cpu().detach().numpy()
            labels_all.extend(labels.cpu().detach().numpy())
            predict_all.extend(outputs)

            if global_batch % 100 == 0:
                train_auc = roc_auc_score(labels_all,predict_all)
                labels_all,predict_all = [],[]

                # dev 数据
                dev_auc,dev_loss = 0,0
                improve = ''
                if dev_iter is not None:
                    dev_auc,dev_loss = model_evaluate(config,model,dev_iter)

                    if dev_auc > dev_best_auc:
                        dev_best_acc = dev_acc
                        last_batch_improve = global_batch
                        last_epoch_improve = epoch
                        best_model = copy.deepcopy(model)

                msg = 'Epoch:{},Iter:{}/{},Train_loss:{},Train_auc:{},Dev_loss:{},Dev_auc:{},Time:{}'
                now = strftime("%Y-%m-%d %H:%M:%S", localtime())
                logging.info(msg.format(epoch,global_batch,t_total,loss.cpu().data.item(),
                                        train_auc,dev_loss.cpu().data.item(),dev_auc,now))
    return best_model









def model_evaluate(config,model,dev_iter):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    total_inputs_error = []

    with torch.no_grad():
        for batch,(input_ids,token_type_ids,attention_mask,label) in enumerate(data_iter):
            input_ids = input_ids.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = label.to(config.device)
            outputs,loss = model(input_ids, token_type_ids, attention_mask, labels)
            loss_total = loss_total+loss
            predict_all.extend(outputs.cpu().detach().numpy())
            labels_all.extend(labels.cpu().detach().numpy())
    dev_auc = roc_auc_score(labels_all,predict_all)
    return dev_auc,loss_total/len(dev_iter)

def model_save(config, model, num=0, name=None):
    if not os.path.exists(config.save_path[num]):
        os.makedirs(config.save_path[num])
    if name is not None:
        file_name = os.path.join(config.save_path[num], name + '.pkl')
    else:
        file_name = os.path.join(config.save_path[num], config.save_file[num]+'.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved, path: %s", file_name)

