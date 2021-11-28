import os
import json
from src.preprocess.processor import NERProcessor

entitys = ['prov',#省
          'city',#市
          'district',#区
          'devzone',#广义的上的开发区
          'town',#乡镇
          'community',#包含社区、行政村（生产大队、村委会），自然
          'village_group',
          'road',
          'roadno',
          'poi',
          'subpoi',
          'houseno',#楼栋号
          'cellno',#单元号
          'floorno',#楼层号
          'roomno',#房间号
          'detail',#poi 内部的四层关系（house,cell,floor, room）没明确是哪一层，如 xx-xx-x-x，则整体标注 detail。
          'assist',#普通辅助定位词
          'distance',#距离辅助定位词，比如“716 县道北 50 米”中的 50 米
          'intersection',#道路口，交叉口
          'redundant',#非地址元素，如配送提示
          'others'
]


def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_data():
    id = 1
    data = {'O':0}
    for entity in entitys:
        data['B-{}'.format(entity)] = id
        id+=1
        data['I-{}'.format(entity)] = id
        id+=1
        data['E-{}'.format(entity)] = id
        id+=1
    data['S-assist'] = id
    id+=1
    data['S-intersection'] = id
    id+=1
    data['S-poi'] = id
    id+=1
    data['S-district'] = id
    id+=1
    data['S-community'] = id

    save_info('../../data/mid_data',data,'crf_ent2id')

def ready_pretrain_data():
    #外部数据
    # out_data = []
    # with open(file='./data/raw_data/addr_sample',mode='r',encoding='utf-8') as lines:
    #     for line in lines:
    #         if line.strip():
    #             out_data.append(line.strip())

    #复赛数据
    data = []
    processor = NERProcessor('./data/raw_data')
    train_feature,dev_feature,fu_test_feature,_,chu_test_feature = processor.get_data_examples()
    for feature in [train_feature,dev_feature,fu_test_feature,chu_test_feature]:
        for sample in feature:
            data.append(''.join(sample.text).strip())


    return data



if __name__ == '__main__':
    # process_data()
    ready_pretrain_data()