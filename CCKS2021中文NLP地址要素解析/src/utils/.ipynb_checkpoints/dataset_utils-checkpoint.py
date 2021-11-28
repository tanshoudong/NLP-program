import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
from src.utils.evaluator import span_decode
import numpy as np
import random
from src.utils.evaluator import mrc_decode,merge_mrc_predict


class mrc_feature:
    def __init__(self,text,label,pseudo,start_ids,end_ids,query,entity_type):
        self.text = text
        self.label = label
        self.pseudo = pseudo
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.query = query
        self.entity_type = entity_type



class NERDataset(Dataset):
    def __init__(self,train_feature,opt,ent2id):
        self.data = train_feature
        self.nums = len(train_feature)
        self.tokenizer = BertTokenizer(os.path.join(opt.bert_dir, 'vocab.txt'))
        self.ent2id = ent2id
        self.opt = opt


    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self,batch_data):
        max_len = max([len(x.text) for x in batch_data])+2
        input_ids,token_type_ids,attention_mask,labels,raw_text,pseudos = [],[],[],[],[],[]
        start_ids,end_ids = [],[]
        for sample in batch_data:
            text = sample.text
            label = sample.label
            if sample.pseudo:
                pseudos.append(1)
            else:
                pseudos.append(0)
            encode_dict = self.tokenizer.encode_plus(text=text,
                                                max_length=max_len,
                                                pad_to_max_length=True,
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)
            input_ids.append(encode_dict['input_ids'])
            token_type_ids.append(encode_dict['token_type_ids'])
            attention_mask.append(encode_dict['attention_mask'])
            raw_text.append(text)


            if label and self.opt.task_type == 'crf':
                tmp_label = [self.ent2id[x] for x in label]
                tmp_label = [0]+tmp_label+[0]
                if len(tmp_label)<max_len:
                    padding_len = max_len -len(tmp_label)
                    tmp_label = tmp_label+[0]*padding_len
                labels.append(tmp_label)

            if label and self.opt.task_type == 'span':
                start_id,end_id = [0]*len(label),[0]*len(label)
                for i,item in enumerate(label):
                    if 'B' in item or 'S' in item:
                        start_id[i] = self.ent2id[item.strip().split('-')[-1].strip()]
                    if 'E' in item or 'S' in item:
                        end_id[i] = self.ent2id[item.strip().split('-')[-1].strip()]
                #add CLS、SEP
                start_id = [0]+start_id+[0]
                end_id = [0]+end_id+[0]
                #padding
                if len(start_id)<max_len:
                    start_id = start_id + [0]*(max_len-len(start_id))
                    end_id = end_id + [0]*(max_len-len(end_id))
                start_ids.append(start_id)
                end_ids.append(end_id)

                tmp_label = [0] + label + [0]
                if len(tmp_label) < max_len:
                    padding_len = max_len - len(tmp_label)
                    tmp_label = tmp_label + [0] * padding_len
                labels.append(tmp_label)

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).float()
        if self.opt.task_type == 'crf':
            labels = torch.tensor(labels).long()
        pseudos = torch.tensor(pseudos).long()
        start_ids = torch.tensor(start_ids).long()
        end_ids = torch.tensor(end_ids).long()


        if self.opt.task_type == 'crf':
            result = ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'pseudos', 'raw_text']
            return dict(zip(result,[input_ids,token_type_ids,attention_mask,labels,pseudos,raw_text]))
        if self.opt.task_type == 'span':
            result = ['input_ids', 'token_type_ids','attention_mask','labels','pseudos', 'raw_text','start_ids','end_ids']
            return dict(zip(result,[input_ids, token_type_ids, attention_mask,labels, pseudos, raw_text,start_ids,end_ids]))
        
        
    def collate_fn_mrc(self,batch_data):
        max_len = max([len(x.start_ids) for x in batch_data])
        input_ids,token_type_ids,attention_mask,labels,raw_text,pseudos = [],[],[],[],[],[]
        start_ids,end_ids,querys,entitys = [],[],[],[]
        for sample in batch_data:
            text = sample.text
            label = sample.label
            query = sample.query
            start_id = sample.start_ids
            end_id = sample.end_ids
            entity = sample.entity_type
            if sample.pseudo:
                pseudos.append(1)
            else:
                pseudos.append(0)
            encode_dict = self.tokenizer.encode_plus(text=query,
                                                     text_pair=text,
                                                max_length=max_len,
                                                pad_to_max_length=True,
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)
            input_ids.append(encode_dict['input_ids'])
            token_type_ids.append(encode_dict['token_type_ids'])
            attention_mask.append(encode_dict['attention_mask'])
            raw_text.append(text)

            pad_length = max_len - len(start_id)

            start_id = start_id + [0] * pad_length  # CLS SEP PAD都为O
            end_id = end_id + [0] * pad_length
            start_ids.append(start_id)
            end_ids.append(end_id)
            querys.append(query)
            entitys.append(entity)


            if self.opt.task_type == 'mrc' and label:
                tmp_label = [0] + label + [0]
                if len(tmp_label) < max_len:
                    padding_len = max_len - len(tmp_label)
                    tmp_label = tmp_label + [0] * padding_len
                labels.append(tmp_label)

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).float()
        pseudos = torch.tensor(pseudos).long()
        start_ids = torch.tensor(start_ids).long()
        end_ids = torch.tensor(end_ids).long()


        if self.opt.task_type == 'mrc':
            result = ['input_ids', 'token_type_ids','attention_mask','labels','pseudos', 'raw_text','start_ids','end_ids','querys','entitys']
            return dict(zip(result,[input_ids, token_type_ids, attention_mask,labels, pseudos, raw_text,start_ids,end_ids,querys,entitys]))



def infer(model,dev_load,opt,device,ent2id):
    f = open(file='./result.txt',mode='w',encoding='utf-8')
    with open(file='./tcdata/final_test.txt',mode='r',encoding='utf-8') as files:
        raw_texts = []
        for line in files:
            raw_texts.append(line.strip())

    id2ent = {v:k for k,v in ent2id.items()}
    model.eval()
    decode_output = []
    with torch.no_grad():
        for batch,batch_data in enumerate(dev_load):
            raw_text = batch_data['raw_text']
            del batch_data['raw_text']
            labels = batch_data['labels']
            del batch_data['labels']

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)

            if opt.task_type == 'crf':
                tmp_decode = model(**batch_data)[0]
                tmp_decode = [sample[1:-1] for sample in tmp_decode]
                decode_output+=tmp_decode

            if opt.task_type == 'span':
                tmp_decode = model(**batch_data)
                start_logits = tmp_decode[0].cpu().numpy()
                end_logits = tmp_decode[1].cpu().numpy()
                for tmp_start_logits, tmp_end_logits,text in zip(start_logits,end_logits,raw_text):
                    tmp_start_logits = tmp_start_logits[1:1 + len(text)]
                    tmp_end_logits = tmp_end_logits[1:1 + len(text)]
                    predict = span_decode(tmp_start_logits,tmp_end_logits,text,id2ent)
                    decode_output.append(predict)


    for text, decode in zip(raw_texts, decode_output):
        tmp_decode_output = " ".join([id2ent[x] if opt.task_type=='crf' else x  for x in decode])
        f.write('{}\n'.format('\u0001'.join([text, tmp_decode_output])))
    f.close()





def gen_mrc_data(train_feature,ent2query,mode):
    data = []
    for sample in train_feature:
        label = sample.label
        text = sample.text
        pseudo = sample.pseudo
        for entity in ent2query.keys():
            start_ids = [0] * len(text)
            end_ids = [0] * len(text)
            query = ent2query[entity].strip().split(' ')
            if mode != 'test':
                for index,item in enumerate(label):
                    if item == 'B-{}'.format(entity):
                        start_ids[index] = 1
                    if item == 'E-{}'.format(entity):
                        end_ids[index]  =1
                    if item == 'S-{}'.format(entity):
                        start_ids[index] = 1
                        end_ids[index] = 1

            start_ids = [0] + [0] * len(query) + [0] + start_ids + [0]
            end_ids = [0] + [0] * len(query) + [0] + end_ids + [0]
            data.append(mrc_feature(text,label,pseudo,start_ids,end_ids,query,entity))

    if mode == 'train':
        data = mrc_data_balance(data)
    return data


def mrc_data_balance(data):
    positive = []
    negetive = []
    for sample in data:
        if sum(sample.start_ids)>0:
            positive.append(sample)
        else:
            negetive.append(sample)

    #负例采样30%，以确保正负样本平衡
    p_nums,n_nums = len(positive),len(negetive)
    negetive = [negetive[i] for i in np.random.choice(a=n_nums,size=p_nums,replace=False)]
    data = negetive+positive
    random.shuffle(data)
    return data



def infer_mrc(model,dev_load,opt,device,ent2id):
    f = open(file='./result.txt',mode='w',encoding='utf-8')
    with open(file='./tcdata/final_test.txt',mode='r',encoding='utf-8') as files:
        raw_char = []
        for line in files:
            raw_char.append(line.strip())

    id2ent = {v:k for k,v in ent2id.items()}
    candidate = {}
    model.eval()
    decode_output = []
    with torch.no_grad():
        for batch,batch_data in enumerate(dev_load):
            raw_text = batch_data['raw_text']
            del batch_data['raw_text']
            querys = batch_data['querys']
            entitys = batch_data['entitys']
            del batch_data['labels']
            del batch_data['querys']
            del batch_data['start_ids']
            del batch_data['end_ids']
            del batch_data['entitys']

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)

            if opt.task_type == 'mrc':
                decode_output = model(**batch_data)
                start_logits = decode_output[0].cpu().numpy()
                end_logits = decode_output[1].cpu().numpy()
                for tmp_start_logits, tmp_end_logits, text, query, entity in zip(start_logits, end_logits,
                                                                                        raw_text,querys,
                                                                                        entitys):

                    tmp_start_logits = tmp_start_logits[2 + len(query):len(text) + len(query) + 2]
                    tmp_end_logits = tmp_end_logits[2 + len(query):len(text) + len(query) + 2]

                    predict = mrc_decode(tmp_start_logits, tmp_end_logits, text, entity)
                    if ''.join(text) in candidate:
                        candidate[''.join(text)].append(predict)
                    else:
                        candidate[''.join(text)] = [predict]

    result = merge_mrc_predict(candidate)
    result = dict(result)

    for line in raw_char:
        id,text = line.strip().split('\u0001')
        text = text.strip()
        label = result[text]
        f.write('{}\n'.format('\u0001'.join([line,' '.join(label)])))
    f.close()
