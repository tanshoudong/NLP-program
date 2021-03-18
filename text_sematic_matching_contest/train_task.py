#coding utf-8
import os
import logging
import torch
from time import strftime, localtime
import random
import time
from torch.utils.data import DataLoader
from utils import set_seed,Vocab,model_save,model_evaluate
from transformers import BertModel,BertConfig,BertForMaskedLM,AdamW,BertTokenizer,BertForSequenceClassification
import torch.distributed as dist
from torch.utils.data import Dataset
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
        self.learning_rate = 1e-5                  # 学习率
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


class BuildDataSet(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        self.tokenizer=BertTokenizer.from_pretrained('vocab')
        self._len = len(self.dataset)

    def __getitem__(self, index):

        text,label=self.dataset[index]
        inputs = self.tokenizer(text)
        return (inputs,label)

    def __len__(self):
        return self._len


def collate_fn(batch_data,padding_token=0,pad_token_segment_id=0):
    max_len = max([len(x[0]['input_ids']) for x in batch_data])
    input_ids, token_type_ids, attention_mask, label=[],[],[],[]
    for x,y in batch_data:
        input_ids.append(x['input_ids']+(max_len-len(x['input_ids']))*[padding_token])
        token_type_ids.append(x['token_type_ids']+(max_len-len(x['token_type_ids']))*[pad_token_segment_id])
        attention_mask.append(x['attention_mask']+(max_len-len(x['attention_mask']))*[0])
        label.append(int(y))

    input_ids = torch.tensor(data=input_ids).type(torch.LongTensor)
    token_type_ids = torch.tensor(data=token_type_ids).type(torch.LongTensor)
    attention_mask = torch.tensor(data=attention_mask).type(torch.LongTensor)
    label = torch.tensor(data=label).type(torch.LongTensor)
    return input_ids, token_type_ids, attention_mask, label


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def train(config):
    vocab=Vocab(config)
    # vocab.add_words()
    # vocab.build_bert_vocab()
    train_data,valid_data,test_data = vocab.get_train_dev_test()


    # 3）使用DistributedSampler

    train_dataset = BuildDataSet(train_data)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_load = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn,sampler=train_sampler)

    valid_dataset = BuildDataSet(valid_data)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_load = DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn,sampler=valid_sampler)

    test_dataset = BuildDataSet(test_data)
    test_load = DataLoader(dataset=test_dataset,batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn)


    #load source bert weights
    model_config = BertConfig.from_pretrained(pretrained_model_name_or_path="bert_source/bert_config.json")
    model = BertForSequenceClassification(config=model_config)
    model.load_state_dict(torch.load('save_bert/checkpoint.pth.tar'),strict=False)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

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

    best_dev_auc = 0
    model.train()
    for epoch in range(config.num_train_epochs):
        # train_sampler.set_epoch(epoch)

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

        dev_auc = model_evaluate(config, model, valid_load)

        if config.local_rank in [0, -1]:
            now = strftime("%Y-%m-%d %H:%M:%S", localtime())
            msg = "time:{},epoch:{}/{},dev_auc:{},best_dev_auc:{}"
            print(msg.format(now, epoch + 1,config.num_train_epochs,dev_auc,best_dev_auc)
            if dev_auc > best_dev_auc:
                best_dev_auc = dev_auc
                torch.save(model.module.state_dict(), 'save_bert' + os.sep + 'best_model.pth.tar')

    #predict test_file



if __name__ == '__main__':
    config=roBerta_Config()
    config.local_rank = local_rank
    config.device = device
    config.nprocs = torch.cuda.device_count()

    set_seed(config)
    train(config)

