#coding utf-8
import os
import torch
from time import strftime, localtime
import pandas as pd
import random
import time
from torch.utils.data import DataLoader
from utils import set_seed,Vocab,submit_result
from transformers import BertModel,BertConfig,BertForMaskedLM,AdamW,BertTokenizer,BertForSequenceClassification
import torch.distributed as dist
from torch.utils.data import Dataset
import pandas as pd
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler


def model_infer(config,test_load,k):
    
    print("***********load model weight*****************")

    model_config = model_config = BertConfig()
    model_config.vocab_size = len(pd.read_csv('../user_data/vocab',names=["score"]))
    
    model = BertForSequenceClassification(config=model_config)
    model.load_state_dict(torch.load('../user_data/save_model/{}_best_model.pth.tar'.format(config.model_name))['status'])
    model = model.to(config.device)

    print("***********make predict for test file*****************")

    
    model.eval()
    predict_all = []

    with torch.no_grad():
        for batch, (input_ids, token_type_ids, attention_mask, label) in enumerate(test_load):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            token_type_ids = token_type_ids.to(config.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            logits = outputs.logits
            pred_pob = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            predict_all.extend(list(pred_pob.detach().cpu().numpy()))
    
#     submit_result(predict)
    if k==0:
        df=pd.DataFrame(predict_all,columns=["{}_socre".format(k+1)])
        df.to_csv('./{}_result.csv'.format(config.model_name),index=False)
    else:
        df=pd.read_csv('./{}_result.csv'.format(config.model_name))
        df["{}_socre".format(k+1)] = predict_all
        df.to_csv('./{}_result.csv'.format(config.model_name),index=False)
    
    print("***********done*****************")
    



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







