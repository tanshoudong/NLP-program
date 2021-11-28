import time
import os
import json
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.utils.trainer import train
from src.utils.options import Args
from src.utils.model_utils import build_model
from src.utils.dataset_utils import NERDataset
from src.utils.evaluator import crf_evaluation, span_evaluation
from src.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel
from src.preprocess.processor import NERProcessor
from src.utils.functions_utils import generate_pseudos
from src.utils.functions_utils import ensemble_infer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def train_base(opt, train_feature,dev_feature=None,test_feature=None):
    #加载实体映射表
    with open(os.path.join(opt.mid_data_dir, f'{opt.task_type}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)


    train_dataset = NERDataset(train_feature,opt,ent2id)
    dev_dataset = NERDataset(dev_feature,opt,ent2id)
    test_dataset = NERDataset(test_feature,opt,ent2id)


    if opt.task_type == 'crf':
        model = build_model('crf', opt.bert_dir,opt, num_tags=len(ent2id),
                            dropout_prob=opt.dropout_prob)
    elif opt.task_type == 'mrc':
        model = build_model('mrc', opt.bert_dir,opt,
                            dropout_prob=opt.dropout_prob,
                            use_type_embed=opt.use_type_embed,
                            loss_type=opt.loss_type)
    else:
        model = build_model('span', opt.bert_dir,opt, num_tags=len(ent2id)+1,
                            dropout_prob=opt.dropout_prob,
                            loss_type=opt.loss_type)

    model,device=train(opt, model, train_dataset,dev_dataset,test_dataset,ent2id)
    return model,device

    

def training(opt):
    processor = NERProcessor(opt.raw_data_dir)
    print("开始单折训练和预测：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))

    train_feature,dev_feature,fu_test_feature,pseudo_feature,chu_test_feature = processor.get_data_examples()
    train_feature = train_feature + pseudo_feature
    train_feature = train_feature + dev_feature

    train_base(opt, train_feature,dev_feature,fu_test_feature)
    
    print("结束单折训练和预测：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))


def stacking(opt):
    processor = NERProcessor(opt.max_seq_len)
    train_feature,dev_feature,fu_test_feature,pseudo_feature,chu_test_feature = processor.get_data_examples()
    train = train_feature + dev_feature
    if opt.cv_infer:
        print("开始进行cv训练和推理：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
        test_feature = fu_test_feature
    else:
        print("开始cv生成pseudo数据：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
        test_feature = fu_test_feature + chu_test_feature



    kf = KFold(5, shuffle=True, random_state=42)

    base_output_dir = opt.output_dir
    models = []
    for i, (train_ids, dev_ids) in enumerate(kf.split(train)):
        logger.info(f'Start to train the {i} fold')
        train_raw_examples = [train[_idx] for _idx in train_ids]
        dev_raw_examples = [train[_idx] for _idx in dev_ids]
        # add pseudo data to train data
        train_raw_examples = train_raw_examples + pseudo_feature
        tmp_output_dir = os.path.join(base_output_dir, f'v{i}')

        opt.output_dir = tmp_output_dir
        opt.cv_num = i

        model,device=train_base(opt, train_raw_examples, dev_raw_examples, test_feature)
        models.append(model)

    #生成pseudo数据集
    if opt.cv_infer:
        #ensemble_infer(opt,test_feature,models,device)
        generate_pseudos(opt)
        print("结束cv训练和推理：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    else:
        generate_pseudos(opt)
        print("结束cv生成pseudo数据：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))


if __name__ == '__main__':

    args = Args().get_parser()

    assert args.task_type in ['crf', 'span', 'mrc']

    args.output_dir = os.path.join(args.output_dir, args.task_type)

    set_seed(args.seed)

    if args.attack_train != '':
        args.output_dir += f'_{args.attack_train}'

    if args.weight_decay:
        args.output_dir += '_wd'

    if args.use_fp16:
        args.output_dir += '_fp16'

    if args.task_type == 'span':
        args.output_dir += f'_{args.loss_type}'

    if args.task_type == 'mrc':
        args.output_dir += f'_{args.loss_type}'

    args.output_dir += f'_{args.task_type}'

    if args.mode == 'stack':
        args.output_dir += '_stack'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f'{args.mode} {args.task_type}')

    if args.mode == 'train':
        training(args)
    else:
        stacking(args)
