#coding utf-8
import os
import logging
import torch
from time import strftime, localtime
import random
import time
from torch.utils.data import DataLoader
from utils import set_seed,Vocab,BuildDataSet,model_save,collate_fn
from transformers import BertModel,BertConfig,BertForMaskedLM,AdamW
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler

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
        self.data_dir=os.getcwd()+os.sep+'data'+os.sep+ \
                      os.path.join('Preliminary', 'gaiic_track3_round1_train_20210228.tsv')
        self.embed_dir=os.getcwd()+os.sep+'data'+ os.sep + 'vector'
        self.models_name = 'roberta'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dropout_prob = 0.1
        self.num_train_epochs = 100                  # epoch数
        self.batch_size = 128                       # mini-batch大小
        self.learning_rate = 2e-5                  # 学习率
        self.head_learning_rate = 1e-3             # 后面的分类层的学习率
        self.weight_decay = 0.01                   # 权重衰减因子
        self.warmup_proportion = 0.1               # Proportion of training to perform linear learning rate warmup for.
        # logging
        # save
        self.load_save_model = False
        self.save_path = [os.getcwd()+os.sep+'data'+os.sep + 'model_data']
        self.save_file = [self.models_name]
        self.seed = 12345
        # 差分学习率
        self.diff_learning_rate = False
        # prob
        self.n_gpu = torch.cuda.device_count()
        self.vocab_size = None
        self.data_enhance = True
        self.embeding_size = 128
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

    # 3）使用DistributedSampler

    train_dataset = BuildDataSet(train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_load = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn,sampler=train_sampler)

    #load source bert weights
    model_config = BertConfig.from_pretrained(pretrained_model_name_or_path="bert_source/bert_config.json")
    model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path="bert_source", config=model_config)

    # model_config = BertConfig()
    # model = BertForMaskedLM(config=model_config)


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
            reduced_loss = reduce_mean(loss, config.nprocs)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        if config.local_rank in [0, -1]:
            now = strftime("%Y-%m-%d %H:%M:%S", localtime())
            print("time:{},epoch:{}/{},mlm_reduce_loss:{},loss:{}".format(now, epoch + 1, config.num_train_epochs,
                                                                          reduced_loss.item(), loss.item()))
            torch.save(model.module.state_dict(), 'save_bert' + os.sep + 'checkpoint.pth.tar')


if __name__ == '__main__':
    config=roBerta_Config()
    config.local_rank = local_rank
    config.device = device
    config.nprocs = torch.cuda.device_count()

    set_seed(config)
    pre_trained(config)

