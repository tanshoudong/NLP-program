#coding utf-8
import os
import torch
from time import strftime, localtime
import random
import time
from torch.cuda.amp import autocast as ac
import numpy as np
from torch.utils.data import DataLoader
from src.utils.functions_utils import set_seed
from transformers import BertModel,BertConfig,BertForMaskedLM,AdamW
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import sys
from transformers import  BertTokenizer
from src.preprocess.convert_raw_data import ready_pretrain_data
from src.utils.modeling_nezha import NeZhaForMaskedLM
from torch.utils.data import Dataset
import time

#单机多进程多卡分布式训练
# 1) 初始化
# torch.distributed.init_process_group(backend="nccl")

# 2） 配置每个进程的gpu
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)

class bert_config:
    def __init__(self):
        #数据路径
        self.data_dir='./data/raw_data'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_train_epochs = 10                  # epoch数
        self.batch_size = 128                       # mini-batch大小
        self.learning_rate = 2e-5                  # 学习率
        self.model_name=None             # bert
        self.weight_decay = 0.01                   # 权重衰减因子
        self.warmup_proportion = 0.1               # Proportion of training to perform linear learning rate warmup for.
        self.seed = 12345
        # 差分学习率
        self.diff_learning_rate = False
        # prob
        self.n_gpu = torch.cuda.device_count()
        self.vocab_size = None
        self.max_grad_norm = 1
        self.use_fp16 = True

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class BuildDataSet(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        self._len = len(self.dataset)
        self.vacab = list(set(list(''.join(dataset))))


    def __getitem__(self, index):
        example=self.dataset[index]
        example,label = random_mask(example,self.vacab)
        # inputs = self.tokenizer(" ".join(example), return_tensors="pt")
        # label = self.tokenizer(" ".join(label))["input_ids"]
        return (example,label)

    def __len__(self):
        return self._len


def random_mask(text_ids,vacab):
    """随机mask
    """
    input, output = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input.append('[MASK]')
            output.append(i)
        elif r < 0.15 * 0.9:
            input.append(i)
            output.append(i)
        elif r < 0.15:
            input.append(vacab[np.random.choice(len(vacab))])
            output.append(i)
        else:
            input.append(i)
            output.append('[PAD]')
    return input, output


def collate_fn(batch_data):
    tokenizer = BertTokenizer('./data/bert/nezha-base-www/vocab.txt')
    max_len = max([len(x[0]) for x in batch_data]) + 2
    input_ids, token_type_ids, attention_mask, labels = [], [], [], []
    for text,label in batch_data:
        inputs = tokenizer.encode_plus(text=text,
                                                 max_length=max_len,
                                                 pad_to_max_length=True,
                                                 is_pretokenized=True,
                                                 return_token_type_ids=True,
                                                 return_attention_mask=True,truncation=True)
        label = tokenizer.encode_plus(text=label,
                                            max_length=max_len,
                                            pad_to_max_length=True,
                                            is_pretokenized=True,
                                            return_token_type_ids=False,
                                            return_attention_mask=False,truncation=True)
        input_ids.append(inputs['input_ids'])
        token_type_ids.append(inputs['token_type_ids'])
        attention_mask.append(inputs['attention_mask'])
        labels.append(label['input_ids'])
    input_ids = torch.tensor(input_ids).long()
    token_type_ids = torch.tensor(token_type_ids).long()
    attention_mask = torch.tensor(attention_mask).float()
    labels = torch.tensor(labels).long()
    return input_ids, token_type_ids, attention_mask, labels




def pre_trained(config):
    train = ready_pretrain_data()[:1000]
    print("pretrain data nums:{}".format(len(train)))

    # 3）使用DistributedSampler

    train_dataset = BuildDataSet(train)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_load = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=collate_fn)

    #load source bert weight
    path = './data/bert/nezha-base-www/'
    model = NeZhaForMaskedLM.from_pretrained(path,output_hidden_states=True,hidden_dropout_prob=0.1)

    for param in model.parameters():
        param.requires_grad = True
    
    scaler = None
    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    # 4) 封装之前要把模型移到对应的gpu
    model = model.to(config.device)

    no_decay = ["bias", "LayerNorm.weight"]

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
    cudnn.benchmark = True

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5)封装
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[config.local_rank],find_unused_parameters=True)

    model.train()
    for epoch in range(config.num_train_epochs):
        # train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()

        for batch, (input_ids, token_type_ids, attention_mask, label) in enumerate(train_load):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            label = label.to(config.device)

            if config.use_fp16:
                with ac():
                    loss = model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids, labels=label)[0]
            else:
                loss = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=label)[0]

            #同步各个进程的速度,计算分布式loss
            # torch.distributed.barrier()
            # if torch.cuda.device_count() > 1:
            #   reduced_loss = reduce_mean(loss, config.nprocs)
            # else:
            #   reduced_loss = loss
            
            model.zero_grad()
            if config.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if config.use_fp16:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            if config.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            if batch%100:
                torch.cuda.empty_cache()
            
        # if config.local_rank in [0, -1]:
        now = strftime("%Y-%m-%d %H:%M:%S", localtime())
        print("time:{},epoch:{}/{},loss:{}".format(now,epoch+1,config.num_train_epochs,loss.item()))
        if torch.cuda.device_count() > 1:
            checkpoint=model.module.state_dict()
        else:
            checkpoint=model.state_dict()
        torch.save(checkpoint,'./data/pre_bert/nezha-base-www/pytorch_model.bin')
        del checkpoint


if __name__ == '__main__':
    print("模型预训练开始：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    config=bert_config()
    set_seed(config.seed)
    pre_trained(config)
    print("模型预训练结束：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))

