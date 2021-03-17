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
from transformers import  RobertaTokenizer

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


def train_dev_test_for_mlm(train,dev,test):
    df = pd.concat([dev,test])
    df["label"] = -1
    tmp = pd.DataFrame()
    tmp["text_left"], tmp["text_right"], tmp["label"] = df["text_right"], df["text_left"], df["label"]
    all = pd.concat([train,df,tmp])
    return all


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



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
        with open(file='vocab', mode="w", encoding="utf-8") as f:
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
                    # examples.append(text_b + text_a)

        return examples



class BuildDataSet(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        self.tokenizer=BertTokenizer.from_pretrained('vocab')
        self._len = len(self.dataset)

    def __getitem__(self, index):

        example=self.dataset[index]
        example,label = random_mask(example)
        inputs = self.tokenizer(" ".join(example), return_tensors="pt")
        label = self.tokenizer(" ".join(label))["input_ids"]
        return inputs,label

    def __len__(self):
        return self._len



class BuildDataSet_v1(Dataset):
    def __init__(self,dataset,vocab):
        self.dataset = dataset.values
        self.vocab = vocab
        self._len = len(self.dataset)

    def __getitem__(self, index):
        example=self.dataset[index]
        text_a=example[0].strip().split(" ")
        text_b=example[1].strip().split(" ")
        label=example[2]
        # text_a = [self.vocab[x] for x in text_a]
        # text_b = [self.vocab[x] for x in text_b]

        return text_a + text_b,label

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
        if r < 0.15 * 0.8:
            input_ids.append('[MASK]')
            output_ids.append(i)
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15:
            input_ids.append(str(np.random.choice(20600)))
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append('[PAD]')
    return input_ids, output_ids





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

    # global_batch = 0  # 记录进行到多少batch
    dev_best_auc = 0
    best_epoch = 0
    # predict_all = []
    # labels_all = []
    best_model = copy.deepcopy(model)

    if config.n_gpu>1:
        model = torch.nn.DataParallel(model)

    for epoch in range(config.num_train_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))

        for batch, (inputs,attention_mask,output_id,label) in enumerate(train_iter):
            model.train()
            inputs = inputs.to(config.device)
            attention_mask = attention_mask.to(config.device)
            output_id = output_id.to(config.device)
            labels = label.to(config.device)
            loss,_ = model(inputs,attention_mask,output_id,label)
            if config.n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

        dev_auc,dev_loss = model_evaluate_v1(config,model,dev_iter)
        if dev_auc > dev_best_auc:
            dev_best_auc = dev_auc
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model)
        msg = 'Epoch:{}/{},Dev_loss:{},Dev_auc:{},best_dev_auc:{},Time:{}'
        now = strftime("%Y-%m-%d %H:%M:%S", localtime())
        logging.info(msg.format(epoch+1,config.num_train_epochs,dev_loss,dev_auc,dev_best_auc,now))

            # outputs = outputs.cpu().detach().numpy()
            # labels_all.extend(labels.cpu().detach().numpy())
            # predict_all.extend(outputs)

            # if global_batch % 100 == 0:
            #     train_auc = roc_auc_score(labels_all,predict_all)
            #     labels_all,predict_all = [],[]

                # dev 数据
                # dev_auc,dev_loss = 0,0
                # if dev_iter is not None:
                #     dev_auc,dev_loss = model_evaluate_v1(config,model,dev_iter)
                #
                #     if dev_auc > dev_best_auc:
                #         dev_best_auc = dev_auc
                #         last_batch_improve = global_batch
                #         last_epoch_improve = epoch
                #         best_model = copy.deepcopy(model)

                # msg = 'Epoch:{},Iter:{}/{},Train_loss:{},Train_auc:{},Dev_loss:{},Dev_auc:{},Time:{}'
                # now = strftime("%Y-%m-%d %H:%M:%S", localtime())
                # logging.info(msg.format(epoch,global_batch,t_total,loss.cpu().data.item(),
                #                         train_auc,dev_loss.cpu().data.item(),dev_auc,now))
    return best_model



def model_evaluate(config,model,data_iter,test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    # total_inputs_error = []

    with torch.no_grad():
        for batch,(input_ids,token_type_ids,attention_mask,label) in enumerate(data_iter):
            input_ids = input_ids.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = label.to(config.device)
            outputs,loss = model(input_ids, token_type_ids, attention_mask, labels)
            if config.n_gpu > 1:
                loss = loss.mean()
            loss_total = loss_total+loss
            predict_all.extend(outputs.cpu().detach().numpy())
            labels_all.extend(labels.cpu().detach().numpy())


    if test:
        return predict_all
    else:
        dev_auc = roc_auc_score(labels_all, predict_all)
        return dev_auc, loss_total/len(data_iter)

def model_evaluate_v1(config,model,data_iter,test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    # total_inputs_error = []

    with torch.no_grad():
        for batch, (inputs,attention_mask,output_id,label) in enumerate(data_iter):
            inputs = inputs.to(config.device)
            attention_mask = attention_mask.to(config.device)
            output_id = output_id.to(config.device)
            labels = label.to(config.device)
            loss,outputs = model(inputs,attention_mask,output_id,label)
            if config.n_gpu > 1:
                loss = loss.mean()
            loss_total = loss_total+loss
            predict_all.extend(outputs.cpu().detach().numpy())
            labels_all.extend(labels.cpu().detach().numpy())


    if test:
        return predict_all
    else:
        dev_auc = roc_auc_score(labels_all, predict_all)
        return dev_auc, loss_total/len(data_iter)


def model_save(config, model, num=0, name=None):
    if not os.path.exists(config.save_path[num]):
        os.makedirs(config.save_path[num])
    if name is not None:
        file_name = os.path.join(config.save_path[num], name + '.pkl')
    else:
        file_name = os.path.join(config.save_path[num], config.save_file[num]+'.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved, path: %s", file_name)


def submit_result(ls):
    with open(file='result.tsv',mode="w",encoding="utf-8") as f:
        for line in ls:
            f.write(str(line)+'\n')



def model_load(config, model, num=0, device='cpu'):
    file_name = os.path.join(config.save_path[num], config.save_file[num]+'.pkl')
    logger.info('loading model: %s', file_name)
    model = model.load_state_dict(torch.load(file_name,
                                     map_location=device if device == 'cpu' else "{}:{}".format(device, 0)))
    return model