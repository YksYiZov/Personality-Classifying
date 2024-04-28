# Personality-Classifying
The Project of NLP By Xinyu Dai

(本项目已经上传github，[我的仓库](https://github.com/YksYiZov/Personality-Classifying/tree/task-2))
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
(如果目录结构有修改需要自行更改代码中的路径，默认预训练模型和数据集在根目录下)
运行Analize.ipynb即可。

