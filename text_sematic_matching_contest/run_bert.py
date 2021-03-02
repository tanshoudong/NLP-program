import os
import logging
import torch
import time
from utils import set_seed,DataProcessor,train_vector
from models import Bert



class Bert_Config:
    def __init__(self):
        #数据路径
        self.data_dir = 'D:/text_sematic_matching_contest/data/Preliminary/gaiic_track3_round1_train_20210228.tsv'
        self.task = 'TianChi'
        self.embed_dir='./data/vector'
        self.models_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.requires_grad = True
        self.class_list = []
        self.num_labels = 2
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = [768]
        self.early_stop = False
        self.require_improvement = 500
        self.num_train_epochs = 5                  # epoch数
        self.batch_size = 64                       # mini-batch大小
        self.learning_rate = 2e-5                  # 学习率
        self.head_learning_rate = 1e-3             # 后面的分类层的学习率
        self.weight_decay = 0.01                   # 权重衰减因子
        self.warmup_proportion = 0.1               # Proportion of training to perform linear learning rate warmup for.
        self.k_fold = 5
        self.multi_drop = 5
        # logging
        self.is_logging2file = True
        self.logging_dir = './data/log' + '/' + self.models_name
        # save
        self.load_save_model = False
        self.save_path = ['./data/model_data']
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

def bert_task(config):
    processor = DataProcessor(config)
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)
    train_examples = processor.get_train_examples()
    train, valid=processor.train_dev_split(train_examples)
    model=Bert(config)










if __name__ == '__main__':
    config=Bert_Config()
    tmp = train_vector(config)
    # set_seed(config)
    # logging_filename = None
    #
    # if config.is_logging2file is True:
    #     file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
    #     logging_filename = os.path.join(config.logging_dir, file)
    #     if not os.path.exists(config.logging_dir):
    #         os.makedirs(config.logging_dir)
    #
    # logging.basicConfig(filename=logging_filename, format='%(levelname)s: %(message)s', level=logging.INFO)
    # logging.info("config %s",config.__dict__)
    # bert_task(config)

