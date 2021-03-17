from transformers import BertModel,BertConfig,BertForMaskedLM,
import torch
import numpy as np


model_config = BertConfig()
model_config.vocab_size=21128
model = BertForMaskedLM(config=model_config)

config = BertConfig.from_pretrained(pretrained_model_name_or_path="bert_source/bert_config.json")
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path="bert_source",config=config)
torch.save(model.module.state_dict())

# inputs = tokenizer.encode_plus("The capital of France is <mask>.", return_tensors="pt")
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits