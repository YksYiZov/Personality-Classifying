# Personality-Classifying
The Project of NLP By Xinyu Dai

(本项目已经上传github，[我的仓库](https://github.com/YksYiZov/Personality-Classifying/tree/main))
## Task 2
### 子问题1
#### 概述
对于HPR的难点，在实验报告中有详细提到，请参看Report->document
##### 难点一
不同性格之间文本内容构成可能很相似
##### 难点二
分类方法不合理，多种分类可能太细致，让文本上差距很小
### 子问题2
#### 实验流程
- 统计各种性格的数量，从中选择两种完全对立的性格然后比较他们的文本之间的相似度。
- 统计每一个维度的性格的数量，选择对立的性格，然后仅仅只做二分类，观察分类效果。
#### 复现流程
运行Analize.ipynb即可。

可能会影响复现过程的参数如下：
- DataFilePath: 参数意为数据集的路径，如在本项目结构中，目录为personality_dataset，不需要表明训练集和验证集
- EPOCHS: 训练轮次，可根据训练速度灵活调整
- LR: 学习率，本项目采用1e-6作为学习率，可以灵活调整
- number： 选入的训练集和验证集数目，可以根据机器性能进行调整

