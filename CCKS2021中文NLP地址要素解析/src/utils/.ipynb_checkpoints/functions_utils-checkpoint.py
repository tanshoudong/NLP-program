import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict,Counter
from datetime import timedelta
import time
import logging
import pdb

logger = logging.getLogger(__name__)



def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if gpu_ids!='-1' and len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    tmp = os.listdir(base_dir)
    tmp = ['checkpoint_{}_epoch.pt'.format(i) for i in range(21) if 'checkpoint_{}_epoch.pt'.format(i) in tmp]
    model_lists = [os.path.join(base_dir,x) for x in tmp]
    return model_lists


def swa(model, model_dir):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)
    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list:
            logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1
    return swa_model


def generate_pseudos():
    tmp = []
    f1 = open(file='./data/pseudo_data/pseudo.txt',encoding='utf-8',mode='w')
    path = './cv_tmp/temp_result_{}'
    for i in range(5):
        with open(file=path.format(i),mode='r',encoding='utf-8') as f:
            tmp.append(f.readlines())
    for item in zip(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4]):
        head = item[0].strip().split('\u0001')[:2]
        candidate = [x.strip().split('\u0001')[-1].strip() for x in item]
        raw_text = item[0].strip().split('\u0001')[1]
        result = vote_ensemble(candidate, raw_text)
        head.append(result)
        target = "\u0001".join(head)
        f1.write("{}\n".format(target))


def vote_ensemble(candidate,raw_text):
    tags = [x.strip().split(" ") for x in candidate]
    entitys = []
    for line in candidate:
        b,e = 0,0
        tmp_entity = []
        tmp = line.strip().split(" ")
        for id,item in enumerate(tmp):
            if "B" in item:
                b = id
            elif "E" in item:
                e = id
                tmp_entity.append(" ".join(tmp[b:e+1]))
            elif item=='O' or "S" in item:
                tmp_entity.append(tmp[id])
        entitys.append(tmp_entity)

    #实体数目一致，按实体投票，不一致，按照tags投票
    if len(set([len(ent) for ent in entitys]))==1:
        result = entity_vote(entitys)
    else:
        result = entity_vote(tags)

    if len(result.strip().split(" "))!=len(raw_text):
        result = entity_vote(tags)

    assert len(result.strip().split(" "))==len(raw_text)

    return result


def entity_vote(entitys):
    result = []
    for i in range(len(entitys[0])):
        cout = Counter([item[i] for item in entitys])
        most_pro = cout.most_common(1)[0][0]
        result.append(most_pro)

    return " ".join(result)