### 英特尔创新大师杯”深度学习挑战赛  赛道2：CCKS2021中文NLP地址要素解析-rank4
    团队名称：spring
    赛题地址：https://tianchi.aliyun.com/competition/entrance/531900/introduction


### 代码结构
    ├── competition_predict.py  #废弃文件
    ├── convert_test_data.py    #废弃文件
    ├── data    #存储数据文件、模型权重和中间文件
    │   ├── bert    #存储bert模型原始权重
    │   │   ├── nezha-base-www  #nezha模型原始权重，复赛最优成绩使用该模型
    │   │   │   ├── config.json
    │   │   │   ├── pytorch_model.bin
    │   │   │   └── vocab.txt
    │   │   └── roberta-base-wwm    #roberta原始权重，复赛最优成绩未使用该模型
    │   │       ├── config.json
    │   │       ├── pytorch_model.bin
    │   │       └── vocab.txt
    │   ├── mid_data    #分别对应crf、span、mrc任务的配置文件
    │   │   ├── crf_ent2id.json
    │   │   ├── mrc_ent2id.json
    │   │   ├── pretrain_data   #废弃文件，复赛未使用
    │   │   └── span_ent2id.json
    │   ├── pre_bert    #复赛预训练模型权重存储目录
    │   │   ├── nezha-base-www
    │   │   │   ├── config.json
    │   │   │   └── vocab.txt
    │   │   └── roberta-base-wwm
    │   │       ├── config.json
    │   │       └── vocab.txt
    │   ├── pseudo_data #伪标签存储目录
    │   │   └── pseudo.txt  #初始为空文件，运行过程中会产生伪标签数据
    │   └── raw_data    #初赛官方的训练、验证、测试数据
    │       ├── addr_sample #废弃数据
    │       ├── dev.conll
    │       ├── final_test.txt
    │       ├── pseudo.txt  #废弃数据
    │       └── train.conll
    ├── Dockerfile  #镜像文件
    ├── main_mrc.py #mrc任务入口文件
    ├── main.py #crf、span任务入口文件
    ├── pretrain_bert.py    #预训练入口文件
    ├── run.sh  #全流程入口文件
    ├── src #源码
    │   ├── preprocess  #部分预处理文件夹
    │   │   ├── convert_raw_data.py
    │   │   └── processor.py
    │   └── utils   #工具文件夹
    │       ├── attack_train_utils.py   #pgd、fgm对抗训练工具函数
    │       ├── configuration_nezha.py  #nezha模型源文件
    │       ├── conlleval.py    #评估脚本 for CoNLL'00
    │       ├── dataset_utils.py    #自定义数据函数
    │       ├── e-b-sequence.txt    #废弃文件
    │       ├── evaluator.py    #部分评估函数
    │       ├── functions_utils.py  #部分工具函数
    │       ├── modeling_nezha.py
    │       ├── model_utils.py  #模型函数
    │       ├── options.py  #全局配置文件
    │       ├── test.py #废弃文件
    │       └── trainer.py  #训练函数
    └── tcdata  #用于本地调试和模拟复赛线上运行过程，数据为初赛数据
        ├── dev.conll
        ├── final_test.txt
        └── train.conll

### 训练流程说明
    全流程分为三个阶段：预训练、迭代伪标降噪和训练推理。
    1.预训练模型为nezha-base-www，数据使用初赛训练集、验证集、测试集和复赛测试集(线上读取)，动态mask；
    2.五折交叉验证对初赛和复赛测试集进行预测，并且投票融合，融合过程中进行过滤，生成伪标签数据，如此为一个循环，循环3次，生成准确的伪标签数据；
    3.伪标签数据生成后，五折交叉验证正式进行训练，最后进行预测和投票融合，生成预测结果。

### 特别说明
    全流程只使用了nezha-base-www预训练模型权重，权重下载地址：https://github.com/lonePatient/NeZha_Chinese_PyTorch，
    pytorch权重，nezha-base-wwm 百度网盘链接：https://pan.baidu.com/share/init?surl=itZ_wdU6JdpXx2saK_zQhw 提取码: ysg3
    具体方案细节可以参考答辩ppt
    
### 感受
    参加完决赛答辩之后，发现冠军方案并没有用什么炫酷的技术，都是大家常用的，但是人家数据做的好，数据增强做的好，扩充了一些数据，拿到了好成绩
    所以说，数据为王啊，一些常规的数据处理(数据增强、数据清洗、过滤、降噪、伪标签)还是要好好做的。
    
### 参考
    https://zhuanlan.zhihu.com/p/326302618
    https://github.com/z814081807/DeepNER