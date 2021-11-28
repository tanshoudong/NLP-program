#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.7-cuda11.0-py3
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
##安装python依赖包 
RUN pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple 
# RUN pip install Flask -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pytorch-crf -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt-get update && apt-get install -y curl
## 把当前文件夹里的文件构建到镜像的根目录下,并设置为默认工作目录 
ADD . / 
WORKDIR / 
## 镜像启动后统一执行 sh run.sh 
CMD ["sh", "run.sh"]
