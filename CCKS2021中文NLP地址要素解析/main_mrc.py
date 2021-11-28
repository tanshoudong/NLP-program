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
from src.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, get_time_dif
from src.preprocess.processor import NERProcessor
from src.utils.dataset_utils import gen_mrc_data

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

    train_feature = gen_mrc_data(train_feature,ent2id,'train')
    dev_feature = gen_mrc_data(dev_feature, ent2id, 'dev')
    test_feature = gen_mrc_data(test_feature, ent2id, 'test')

    train_dataset = NERDataset(train_feature,opt,ent2id)
    dev_dataset = NERDataset(dev_feature,opt,ent2id)
    test_dataset = NERDataset(test_feature,opt,ent2id)


    if opt.task_type == 'crf':
        model = build_model('crf', opt.bert_dir, num_tags=len(ent2id),
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

    train(opt, model, train_dataset,dev_dataset,test_dataset,ent2id)

def training(opt):
    processor = NERProcessor(opt.raw_data_dir)

    train_feature,dev_feature,test_feature,pseudo_feature = processor.get_data_examples()
    # train_feature = train_feature + pseudo_feature
    train_feature = train_feature + dev_feature
    train_base(opt, train_feature,dev_feature,test_feature)



def stacking(opt):
    logger.info('Start to KFold stack attribution model')

    if args.task_type == 'mrc':
        # 62 for mrc query
        processor = NERProcessor(opt.max_seq_len-62)
    else:
        processor = NERProcessor(opt.max_seq_len)

    kf = KFold(5, shuffle=True, random_state=42)

    stack_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'stack.json'))

    pseudo_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'pseudo.json'))

    base_output_dir = opt.output_dir

    for i, (train_ids, dev_ids) in enumerate(kf.split(stack_raw_examples)):
        logger.info(f'Start to train the {i} fold')
        train_raw_examples = [stack_raw_examples[_idx] for _idx in train_ids]

        # add pseudo data to train data
        train_raw_examples = train_raw_examples + pseudo_raw_examples
        train_examples = processor.get_examples(train_raw_examples, 'train')

        dev_raw_examples = [stack_raw_examples[_idx] for _idx in dev_ids]
        dev_info = processor.get_examples(dev_raw_examples, 'dev')

        tmp_output_dir = os.path.join(base_output_dir, f'v{i}')

        opt.output_dir = tmp_output_dir

        train_base(opt, train_examples, dev_info)

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
