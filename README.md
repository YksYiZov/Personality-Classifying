# Personality-Classifying
The Project of NLP By Xinyu Dai

## Task 1
### 子问题1
#### 概述
在这个问题中，我们采用了KNN来进行分类任务，具体的执行思路如下
#### 思路
- 导入文本，进行词数统计工作，分析统计的词数，对数据进行预处理
- 将MBTI人格分类任务分为四个子任务，训练四个KNN分类器，每个KNN完成一个方面的二分类任务
#### 实施细节
##### KNN的训练
- 数据的采集工作我们把全部的训练样本分为两类
- 统计两类样本各个词出现的数量
- 根据每个词出现的数量映射到数字特征上
- 使用自编码器或PCA完成降维（或不使用）
- 训练KNN