# coding utf-8
import os
import torch
from time import strftime, localtime
import random
import time
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import set_seed, Vocab, model_evaluate
from transformers import BertModel, BertConfig, BertForMaskedLM, AdamW, BertTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from utils import FGM
from infer import model_infer
from sklearn.model_selection import KFold
import pandas as pd



# 单机多进程多卡分布式训练
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


class roBerta_Config:
    def __init__(self):
        # 数据路径
        self.data_dir = '../data/gaiic_track3_round1_testA_20210228.tsv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dropout_prob = 0.1
        self.num_train_epochs = 3  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.learning_rate = 4e-5  # 学习率
        self.head_learning_rate = 1e-3  # 后面的分类层的学习率
        self.weight_decay = 0.01  # 权重衰减因子
        self.warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for.
        self.seed = 2021
        # 差分学习率
        self.diff_learning_rate = False
        # prob
        self.n_gpu = torch.cuda.device_count()
        self.vocab_size = None
        self.data_enhance = True
        self.embeding_size = 128
        self.max_grad_norm = 1
        self.fgm = True
        self.model_name = ['bert','rbtl']

class BuildDataSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained('../user_data/vocab')
        self._len = len(self.dataset)

    def __getitem__(self, index):
        text, label = self.dataset[index]
        inputs = self.tokenizer(text)
        return (inputs, label)

    def __len__(self):
        return self._len


def collate_fn(batch_data, padding_token=0, pad_token_segment_id=0):
    max_len = max([len(x[0]['input_ids']) for x in batch_data])
    input_ids, token_type_ids, attention_mask, label = [], [], [], []
    for x, y in batch_data:
        input_ids.append(x['input_ids'] + (max_len - len(x['input_ids'])) * [padding_token])
        token_type_ids.append(x['token_type_ids'] + (max_len - len(x['token_type_ids'])) * [pad_token_segment_id])
        attention_mask.append(x['attention_mask'] + (max_len - len(x['attention_mask'])) * [0])
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


def train_process(config, train_load, train_sampler, model_name):
    # load source bert weights
    model_config = BertConfig.from_pretrained(pretrained_model_name_or_path="../user_data/bert_source/{}_config.json".format(model_name))
    # model_config = BertConfig()
    model_config.vocab_size = len(pd.read_csv('../user_data/vocab',names=["score"]))
    model = BertForSequenceClassification(config=model_config)

    checkpoint = torch.load('../user_data/save_bert/{}_checkpoint.pth.tar'.format(model_name),
                                map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['status'], strict=False)
    print('***********load pretrained mlm {} weight*************'.format(model_name))

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

    #     t_total = len(train_load) * config.num_train_epochs
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    #     )

    cudnn.benchmark = True

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5)封装
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank])

    model.train()
    if config.fgm:
        fgm = FGM(model)

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
            model.zero_grad()
            loss.backward()
            #             torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            if config.fgm:
                fgm.attack()  # 在embedding上添加对抗扰动
                loss_adv = model(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, labels=label).loss
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数

            optimizer.step()
        #             scheduler.step()

        # dev_auc = model_evaluate(config, model, valid_load)

        # 同步各个进程的速度,计算分布式loss
        torch.distributed.barrier()
        # reduce_dev_auc = reduce_auc(dev_auc, config.nprocs).item()

        # if reduce_dev_auc > best_dev_auc:
        #     best_dev_auc = reduce_dev_auc
        #     is_best = True

        now = strftime("%Y-%m-%d %H:%M:%S", localtime())
        msg = 'model_name:{},time:{},epoch:{}/{}'

        if config.local_rank in [0, -1]:
            print(msg.format(model_name,now, epoch + 1, config.num_train_epochs))
            checkpoint = {"status": model.module.state_dict()}
            torch.save(checkpoint, '../user_data/save_model' + os.sep + '{}_checkpoint.pth.tar'.format(model_name))
            del checkpoint

    torch.distributed.barrier()

def reduce_auc(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train(config):

    vocab = Vocab(config)
    train_data = vocab.get_train_dev_test()
    train1 = [(x[0] + ' ' + x[1], x[2]) for x in train_data]
    train2 = [(x[1] + ' ' + x[0], x[2]) for x in train_data]
    train_data = train1 + train2
    train_dataset = BuildDataSet(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_load = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn, sampler=train_sampler)

    for model_name in config.model_name:
        if config.local_rank in [0, -1]:
            msg = 'model_name:{},train_nums:{},train_iter:{},batch_size:{}'
            print(msg.format(model_name, len(train_data), len(train_load),config.batch_size))

        train_process(config,train_load,train_sampler,model_name)
        torch.distributed.barrier()






if __name__ == '__main__':
    config = roBerta_Config()
    config.local_rank = local_rank
    config.device = device
    config.nprocs = torch.cuda.device_count()
    set_seed(config)
    train(config)

