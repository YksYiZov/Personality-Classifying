# Personality-Classifying
The Project of NLP By Xinyu Dai

(本项目已经上传github，[我的仓库](https://github.com/YksYiZov/Personality-Classifying/tree/task-3))
## Task 3
### 子问题1
#### 概述
在上一个任务中讨论了MBTI用于文本分类的不太合理的地方，在这一部分，我使用了聚类算法来进行分类的研究，为了能和此前的任务呼应，我依然将样本分为16个类，方便与之前的结果进行比较。
#### 复现流程
在task-3.ipynb中运行代码，直到`train_cluster_id`和`valid_cluster_id`出现，便完成了新的分类标签的设计。
### 子问题2
#### 概述
依托于新的分类标签，重新进行新的实验，借助更小的tiny网络，我可以在本地完成对整个数据集的训练。
#### 复现流程
继续运行task-3.ipynb中的代码，最终会在运行300个训练轮次后终止。注意的是调整`DataFilePath`和`BertModelPath`来匹配自己的数据集与预训练模型位置。或者你也可以直接从网络下载预训练模型。