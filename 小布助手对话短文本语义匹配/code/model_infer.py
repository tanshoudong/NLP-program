#coding utf-8
import os
import torch
from time import strftime, localtime
import pandas as pd
import numpy as np
import onnxruntime
import random
import json
import time
from torch.utils.data import DataLoader
from utils import set_seed,Vocab,submit_result,set_seed
from transformers import BertConfig,BertForMaskedLM,AdamW,BertTokenizer,BertForSequenceClassification
import sys
import time
import requests
from flask import Flask, request
from multiprocessing import Process

app = Flask(__name__)

def make_train_dummy_input():
    org_input_ids = torch.LongTensor([[31, 51, 98, 1]])
    org_token_type_ids = torch.LongTensor([[1, 1, 1, 1]])
    org_input_mask = torch.LongTensor([[0, 0, 1, 1]])
    return (org_input_ids, org_token_type_ids, org_input_mask)

def make_inference_dummy_input():
    inf_input_ids = [[31, 51, 98, 1]]
    inf_token_type_ids = [[1, 1, 1, 1]]
    inf_input_mask = [[0, 0, 1, 1]]
    return (inf_input_ids, inf_token_type_ids, inf_input_mask)

class init_class:
    def __init__(self):
        set_seed()
        self.sess = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('../user_data/vocab')

        for model_name in ['bert','rbtl']:
            model_config = BertConfig.from_pretrained(
                pretrained_model_name_or_path="../user_data/bert_source/{}_config.json".format(model_name))
            model_config.vocab_size = len(pd.read_csv('../user_data/vocab', names=["score"]))

            self.model = BertForSequenceClassification(config=model_config)
            checkpoint = torch.load('../user_data/save_model/{}_checkpoint.pth.tar'.format(model_name), map_location='cpu')
            self.model.load_state_dict(checkpoint['status'])

            #pytorch转onnx
            MODEL_ONNX_PATH = "./torch_{}_dynamic.onnx".format(model_name)
            OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
            self.model.eval()
            org_dummy_input = make_train_dummy_input()
            inf_dummy_input = make_inference_dummy_input()
            dynamic_axes = {'input_ids': [1], 'token_type_ids': [1], 'attention_mask': [1]}
            output = torch.onnx.export(self.model,
                                       org_dummy_input,
                                       MODEL_ONNX_PATH,
                                       verbose=False,
                                       operator_export_type=OPERATOR_EXPORT_TYPE,
                                       opset_version=10,
                                       input_names=['input_ids', 'token_type_ids', 'attention_mask'],
                                       output_names=['output'], dynamic_axes=dynamic_axes)

            self.sess.append(onnxruntime.InferenceSession(MODEL_ONNX_PATH))

    def __getitem__(self,text):
        inputs = self.tokenizer(text,return_tensors="pt")
        result = []
        for sess in self.sess:
            pred_onnx = sess.run(None, {'input_ids': inputs['input_ids'].numpy(),
                                        'token_type_ids': inputs['token_type_ids'].numpy(),
                                        'attention_mask': inputs['attention_mask'].numpy()})

            pred_pob = torch.nn.functional.softmax(torch.tensor(pred_onnx[0]), dim=1)[:, 1]

            result.append(pred_pob[0].cpu().item())
        return np.mean(result)

@app.route("/tccapi", methods=['GET', 'POST'])
def tccapi():
    data = request.get_data()
    if (data == b"exit"):
        print("received exit command, exit now")
        os._exit(0)

    input_list = request.form.getlist("input")
    index_list = request.form.getlist("index")

    response_batch = {}
    response_batch["results"] = []
    for i in range(len(index_list)):
        index_str = index_list[i]

        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split("\t")
            query_A = elems[0].strip()
            query_B = elems[1].strip()
            predict = infer(infer_model, query_A, query_B)
            response["predict"] = predict
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
        response_batch["results"].append(response)

    return json.dumps(response_batch)


# 需要根据模型类型重写
def infer(model, query_A, query_B):
    predict = 0
    query = query_A + " " + query_B
    predict = model[query]
    return predict


# 需要根据模型类型重写
def init_model():
    infer_model = init_class()
    return infer_model


if __name__ == "__main__":
    print(onnxruntime.get_device())
    infer_model = init_model()
    app.run(host="0.0.0.0", port=8080)






