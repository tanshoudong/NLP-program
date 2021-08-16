#coding utf-8
import os
import torch
from time import strftime, localtime
import random
import time
from torch.utils.data import DataLoader
from utils import set_seed,Vocab,BuildDataSet,collate_fn
from transformers import BertModel,BertConfig,BertForMaskedLM,AdamW
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
import sys

#单机多进程多卡分布式训练
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class roBerta_Config:
    def __init__(self):
        #数据路径
        self.data_dir='../data/gaiic_track3_round1_testA_20210228.tsv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_train_epochs = 1                  # epoch数
        self.batch_size = 64                       # mini-batch大小
        self.learning_rate = 2e-5                  # 学习率
        self.model_name=None             # bert,rbtl
        self.weight_decay = 0.01                   # 权重衰减因子
        self.warmup_proportion = 0.1               # Proportion of training to perform linear learning rate warmup for.
        self.seed = 12345
        # 差分学习率
        self.diff_learning_rate = False
        # prob
        self.n_gpu = torch.cuda.device_count()
        self.vocab_size = None
        self.max_grad_norm = 1

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def pre_trained(config):
    vocab=Vocab(config)
    vocab.add_words()
    vocab.build_bert_vocab()
    train = vocab.get_pre_trained_examples()
    print("train nums:{}".format(len(train)))

    # 3）使用DistributedSampler

    train_dataset = BuildDataSet(train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_load = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn,sampler=train_sampler)

    #load source bert weights
    
    model_config = BertConfig.from_pretrained(pretrained_model_name_or_path="../user_data/bert_source/{}_config.json".format(config.model_name))
    model_config.vocab_size = len(pd.read_csv('../user_data/vocab',names=["score"]))
    model = BertForMaskedLM(config=model_config)

#     if os.path.isfile('../user_data/save_bert/bert_checkpoint.pth.tar'):
#         exist_checkpoint = torch.load('../user_data/save_bert/{}_checkpoint.pth.tar'.format(config.model_name),map_location=torch.device('cpu'))
#         exit_status,exit_epoch = exist_checkpoint["status"],exist_checkpoint["epoch"]
#         model = BertForMaskedLM(config=model_config)
#         model.load_state_dict(exit_status)
#         del exit_status
#         print("*********load chechpoin file********")
#     else:
#         model = BertForMaskedLM(config=model_config)
# #         status = torch.load('../user_data/bert_source/{}/pytorch_model.bin'.format(config.model_name),map_location=torch.device('cpu'))
# #         del_ls=['bert.embeddings.word_embeddings.weight','cls.predictions.bias','cls.predictions.decoder.weight','cls.predictions.decoder.bias']
# #         for col in del_ls:
# #             if col in status:
# #                 del status[col]
# #         model.load_state_dict(status,strict=False)
#         exit_epoch = 0
#         print("*********load {}_bert source file********".format(config.model_name))
#         del status

    for param in model.parameters():
        param.requires_grad = True

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
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[config.local_rank])

    model.train()
    for epoch in range(config.num_train_epochs):
        train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()

        for batch, (input_ids, token_type_ids, attention_mask, label) in enumerate(train_load):
            input_ids = input_ids.cuda(config.local_rank, non_blocking=True)
            attention_mask = attention_mask.cuda(config.local_rank, non_blocking=True)
            token_type_ids = token_type_ids.cuda(config.local_rank, non_blocking=True)
            label = label.cuda(config.local_rank, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=label)

            loss = outputs.loss

            #同步各个进程的速度,计算分布式loss
            torch.distributed.barrier()
            if torch.cuda.device_count() > 1:
              reduced_loss = reduce_mean(loss, config.nprocs)
            else:
              reduced_loss = loss

            model.zero_grad()
            loss.backward()
            optimizer.step()
            
        if config.local_rank in [0, -1]:
            now = strftime("%Y-%m-%d %H:%M:%S", localtime())
            print("time:{},epoch:{}/{},mlm_reduce_loss:{}".format(now,epoch+1,config.num_train_epochs,reduced_loss.item()))
            if torch.cuda.device_count() > 1:
              checkpoint={"status":model.module.state_dict(),"epoch":epoch+1}
            else:
              checkpoint={"status":model.state_dict(),"epoch":epoch+1}
            torch.save(checkpoint, '../user_data/save_bert/{}_checkpoint.pth.tar'.format(config.model_name))
            del checkpoint   
#         torch.cuda.empty_cache()


if __name__ == '__main__':
    config=roBerta_Config()
    config.model_name = sys.argv[-1]
    config.local_rank = local_rank
    config.device = device
    config.nprocs = torch.cuda.device_count()

    set_seed(config)
    pre_trained(config)

