#bin/bash
#打印GPU信息 
nvidia-smi 
cd ./code
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 pretrain_bert.py bert
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 pretrain_bert.py rbtl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 finetune.py
python3 model_infer.py
