#coding utf-8
import os
import logging
import torch
from time import strftime, localtime
import random
import time
from torch.utils.data import DataLoader
from utils import set_seed,Vocab,submit_result
from transformers import BertModel,BertConfig,BertForMaskedLM,AdamW,BertTokenizer,BertForSequenceClassification
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler


def model_infer(model,config,data_load):
    model.eval()
    predict_all = []

    with torch.no_grad():
        for batch, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_load):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            token_type_ids = token_type_ids.to(config.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            logits = outputs.logits
            pred_pob = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            predict_all.extend(list(pred_pob.detach().cpu().numpy()))

        return predict_all


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

print("***********load test data*****************")



config = roBerta_Config()
vocab=Vocab()
train_data,valid_data,test_data = vocab.get_train_dev_test()
test_dataset = BuildDataSet(test_data)
test_load = DataLoader(dataset=test_dataset,batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn)


print("***********load model weight*****************")

model_config = BertConfig.from_pretrained(pretrained_model_name_or_path="bert_source/bert_config.json")
model = BertForSequenceClassification(config=model_config)
model.load_state_dict(torch.load('save_bert/best_model.pth.tar'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
config.device = device

print("***********make predict for test file*****************")

predict = model_infer(model,config,test_load)
submit_result(predict)
print("***********done*****************")




