#coding utf-8
import os
import logging
import torch
import time
from torch.utils.data import DataLoader
from utils import set_seed,DataProcessor,train_vector,\
    Vocab,BuildDataSet_v1,train_process,model_save,model_evaluate_v1,submit_result
from models import roBerta



class roBerta_Config:
    def __init__(self):
        #数据路径
        self.data_dir=os.getcwd()+os.sep+'data'+os.sep+ \
                      os.path.join('Preliminary', 'gaiic_track3_round1_train_20210228.tsv')
        self.task = 'TianChi'
        self.embed_dir=os.getcwd()+os.sep+'data'+ os.sep + 'vector'
        self.models_name = 'roberta'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.requires_grad = True
        self.class_list = []
        self.num_labels = 2
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = [512]
        self.num_train_epochs = 8                  # epoch数
        self.batch_size = 128                       # mini-batch大小
        self.learning_rate = 2e-5                  # 学习率
        self.head_learning_rate = 1e-3             # 后面的分类层的学习率
        self.weight_decay = 0.01                   # 权重衰减因子
        self.warmup_proportion = 0.1               # Proportion of training to perform linear learning rate warmup for.
        self.k_fold = 5
        self.multi_drop = 5
        # logging
        self.is_logging2file = True
        self.logging_dir = os.getcwd()+os.sep+'data'+os.sep+os.path.join('log',self.models_name)
        # save
        self.load_save_model = False
        self.save_path = [os.getcwd()+os.sep+'data'+os.sep + 'model_data']
        self.save_file = [self.models_name]
        self.seed = 12345
        # 计算loss的方法
        self.loss_method = 'binary'  # [ binary, cross_entropy]
        # 差分学习率
        self.diff_learning_rate = False
        # train pattern
        self.pattern = 'full_train'   # [only_train, full_train, predict]
        # preprocessing
        self.stop_word_valid = True
        # prob
        self.out_prob = True
        self.n_gpu = torch.cuda.device_count()
        self.vocab_size = None
        self.data_enhance = True
        self.embeding_size = 128
        self.max_grad_norm = 1

def roberta_task(config):
    processor = DataProcessor(config)
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)
    train_examples = processor.get_train_examples()
    train, dev=processor.train_dev_split(train_examples)

    #train 词向量，建立词表
    train_ebeding=train_vector(config)
    vocab=Vocab()
    vocab.add_words(train_ebeding.get_all_exampes_words())
    vocab.build_bert_vocab()

    test = processor.get_test_data(train_ebeding.data_dir[0])

    train_dataset = BuildDataSet_v1(train,vocab)
    train_load = DataLoader(dataset=train_dataset,batch_size=config.batch_size,
                            shuffle=True,collate_fn=train_ebeding.collate_fn_v1)

    dev_dataset = BuildDataSet_v1(dev,vocab)
    dev_load = DataLoader(dataset=dev_dataset,batch_size=config.batch_size,
                        shuffle=True,collate_fn=train_ebeding.collate_fn_v1)
    test_dataset = BuildDataSet_v1(test,vocab)
    test_load = DataLoader(dataset=test_dataset,batch_size=config.batch_size,
                           shuffle=False,collate_fn=train_ebeding.collate_fn_v1)

    config.vocab_size = len(train_ebeding.w2v.wv.vocab) + 5
    model = roBerta(config).to(config.device)

    model_example,last_epoch_improve = train_process(config, model, train_iter = train_load, dev_iter = dev_load)

    model_save(config, model_example)

    predict_result = model_evaluate_v1(config, model_example, test_load,test=True)
    submit_result(predict_result)
    print("best epoch:{}".format(last_epoch_improve))




if __name__ == '__main__':
    config=roBerta_Config()
    set_seed(config)
    logging_filename = None

    if config.is_logging2file is True:
        file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        logging_filename = os.path.join(config.logging_dir, file)
        if not os.path.exists(config.logging_dir):
            os.makedirs(config.logging_dir)

    logging.basicConfig(filename=logging_filename, format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info("config %s",config.__dict__)
    roberta_task(config)

