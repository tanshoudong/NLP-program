import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel,RobertaConfig, RobertaModel
from torch.autograd import Variable
from utils import compute_loss
import logging
from torch.nn import CrossEntropyLoss

class Bert(nn.Module):

    def __init__(self, config, num=0):
        super(Bert, self).__init__()
        model_config = BertConfig()
        model_config.vocab_size = config.vocab_size
        # 计算loss的方法
        self.loss_method = config.loss_method
        self.multi_drop = config.multi_drop

        self.bert = BertModel(model_config)
        if config.requires_grad:
            for param in self.bert.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size[num]
        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.classifier.apply(self._init_weights)
        self.bert.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)


    def forward(self,input_ids=None,token_type_ids=None,attention_mask=None,labels=None):
        outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        out = None
        loss = 0
        for i in range(self.multi_drop):
            output = self.dropout(pooled_output)
            if labels is not None:
                if i == 0:
                    out = self.classifier(output)
                    loss = compute_loss(out, labels, loss_method=self.loss_method)
                else:
                    temp_out = self.classifier(output)
                    temp_loss = compute_loss(temp_out, labels, loss_method=self.loss_method)
                    out = out + temp_out
                    loss = loss + temp_loss

        loss = loss/self.multi_drop
        out = out/self.multi_drop

        if self.loss_method in ['binary']:
            out = torch.sigmoid(out).flatten()

        return out,loss


class roBerta(nn.Module):

    def __init__(self, config, num=0):
        super(roBerta, self).__init__()
        model_config = RobertaConfig()
        model_config.vocab_size = config.vocab_size
        model_config.hidden_size = config.hidden_size[0]
        model_config.num_attention_heads = 16
        # 计算loss的方法
        self.loss_method = config.loss_method
        self.multi_drop = config.multi_drop

        self.roberta = RobertaModel(model_config)
        if config.requires_grad:
            for param in self.roberta.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size[num]
        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.text_linear = nn.Linear(config.embeding_size,config.hidden_size[0])
        self.vocab_layer = nn.Linear(config.hidden_size[0],config.vocab_size)

        self.classifier.apply(self._init_weights)
        self.roberta.apply(self._init_weights)
        self.text_linear.apply(self._init_weights)
        self.vocab_layer.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)


    def forward(self,inputs=None,attention_mask=None,output_id=None,labels=None):
        inputs = torch.relu(self.text_linear(inputs))
        bert_outputs = self.roberta(inputs_embeds=inputs,attention_mask=attention_mask)

        #calculate mlm loss
        last_hidden_state = bert_outputs[0]
        output_id_tmp = output_id[output_id.ne(-100)]
        output_id_emb = last_hidden_state[output_id.ne(-100)]
        pre_score = self.vocab_layer(output_id_emb)
        loss_cro = CrossEntropyLoss()
        mlm_loss = loss_cro(torch.sigmoid(pre_score),output_id_tmp)

        labels_bool = labels.ne(-1)
        if labels_bool.sum().item() == 0:
            return mlm_loss,torch.tensor([])

        #calculate label loss
        pooled_output = bert_outputs[1]
        out = self.classifier(pooled_output)
        out = out[labels_bool]
        labels_tmp = labels[labels_bool]
        label_loss = compute_loss(out,labels_tmp)
        out = torch.sigmoid(out).flatten()
        return mlm_loss + label_loss,out










        return out,loss