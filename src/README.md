### Introduction

本代码为初赛阶段使用的代码，由于时间问题，共使用两个模型来集成最后的结果，复赛阶段两个模型将会融合在一起成为端到端的单模型。

### Requirements

代码基于Pytorch 下的mmdetection, 依赖以下库

- Linux (Windows is not officially supported)
- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC(G++) 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

- OS: Ubuntu 16.04
- CUDA: 9.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 4.9/5.3/5.4/7.3

### Install

a. Create a conda virtual environment and activate it.

```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```
conda install pytorch torchvision -c pytorch
```

c. Install code(other dependencies will be installed automatically).

```
python setup.py develop
```

### Prepare Data

1.确保数据集数据结构如下所示：

|–data

    |-- First_round_data
    
　     |-- guangdong1_round1_train1_20190818.zip
      
　     |-- guangdong1_round1_train2_20190828.zip
      
　     |-- guangdong1_round1_testA_20190818.zip
      

   |-- guangdong1_round1_testB_20190919.zip

2.解压原始数据，原始数据将会解压到data 目录下

./tools/unzip.sh

3.将训练数据和测试数据转化为COCO格式

python ./tools/PrepareData.py

### Train

sh train_model_0.sh 4 

sh train_model_1.sh 4 

分别训练两个模型,默认使用4卡， 修改GPU数量请线性修改学习率

### Test

./tools/inference.sh GPU_NUM

GPU_NUM是推理所使用的GPU数量



### 模型设计

1. 本次比赛为布匹瑕疵检测，使用Cascade RCNN resnext101作为baseline +syncBN + Defromable conv

2. 将正常图片产生的建议区域作为负样本在抽样过程中与正样本混合进行训练，大大提高acc
3. 2中策略由于时间问题目前只在Faster RCNN 中测试并实装，在第一阶段中使用两个模型融合的方式。利用1的预测结果和2中预测出含有瑕疵的图片目录来共同生成最终的结果，同时提高acc 和高map



### 参考资料

雪浪AI挑战赛答辩视频

天池布匹瑕疵挑战赛答辩视频

Cai Z, Vasconcelos N. Cascade r-cnn:Delving into high quality object detection[C]//Proceedings of the IEEE
conference on computer vision and pattern recognition. 2018: 6154-6162.



