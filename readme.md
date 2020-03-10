用于处理日志中参数，提取参数语义。

1.日志数据来源： https://github.com/logpai/logparser。
目前处理的信息数据只有：HDFS,Windows,Zookeeper。

在 https://github.com/logpai/logparser 中有更多日志数据。

2.各个文件：
Wordvector.py: 处理源文件。 run方法生成 input x, output y。
其他.py是各个模型。
目前在这里用于对比的有高斯贝叶斯，随机森林，MLP，KNN，逻辑回归。

data/： 存放日志源文件。

model/:  可存word2vec产生的词向量。

Word_embeddings/: 存放各个日志信息产生的词汇表。

result/: 各个日志对比结果。
