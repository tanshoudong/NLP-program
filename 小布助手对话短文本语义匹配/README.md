### 背景
    全球人工智能技术创新大赛  赛道三: 小布助手对话短文本语义匹配,rank45/5345,后期坑在了docker，复赛截至前三天匆匆提交了一次，
    否则成绩不至于如此.....
   [赛题地址](https://tianchi.aliyun.com/competition/entrance/531851/introduction?spm=5176.12281957.1004.6.72c13eafugruPS)
    
### 算法程序说明
    分为预训练、finetune和infer三个部分，总共三个模型，分别为bert、electra、nezha，每个模型分别进行预训练和finetune，五折交叉验证，结果
    取平均，最终再把三个模型的结果加权融合得到最终结果。
    
### 开源预训练模型来源
    bert：BERT-wwm-base,Chinese，来源：https://github.com/ymcui/Chinese-BERT-wwm
    electra：ELECTRA-base, Chinese，来源：https://github.com/ymcui/Chinese-ELECTRA
    nezha：nezha-base-wwm，来源：https://github.com/lonePatient/NeZha_Chinese_PyTorch
    
### 安装环境
    用Dockerfile打包镜像即可

### 预训练
    训练方式是单机多卡多进程分布式训练
    终端执行：CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 code/pretrain_bert.py
    注：在pretrain_bert.py脚本中修改self.model_name='模型名称' 配置参数，可选bert、electra、nezha，指定预训练哪种模型
    预训练完成后，会在user_data/save_bert/目录中产生文件 模型名称_checkpoint.pth.tar

### finetune
    训练方式是单机多卡多进程分布式训练
    终端执行：CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 code/finetune.py
    注：在kfold.py脚本中修改self.model_name='模型名称' 配置参数，可选bert、electra、nezha，指定finetune哪种模型

### infer
    复赛线上要求提交docker镜像，并且以数据流的形式单条infer，并且单条推理时间不超过15ms，所以，要想融合更多模型得到更好的结果，
    就得使用推理加速的技术了，这次比赛使用了onnx加速，细节参考代码，另外使用flask提供服务，终端执行：python3 model_infer.py
    
### 测试执行
    终端下执行命令：nohup sh run.sh 2>&1 | tee log &

### top方案参考
    前排top选手方案：github地址: https://github.com/daniellibin/gaiic2021_track3_querySim
    
    
    
