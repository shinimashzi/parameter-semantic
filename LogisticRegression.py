# encoding: utf-8

# LogsticRegression
from Wordvector import WordVector
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import time
import os
settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        },

    # 'Hadoop': {
    #     'log_file': 'Hadoop/Hadoop_2k.log',
    #     },
    #
    # 'Spark': {
    #     'log_file': 'Spark/Spark_2k.log',
    #     },
    #
    # 'Zookeeper': {
    #     'log_file': 'Zookeeper/Zookeeper_2k.log',
    #     },
    #
    # 'BGL': {
    #     'log_file': 'BGL/BGL_2k.log',
    #     },
    #
    # 'HPC': {
    #     'log_file': 'HPC/HPC_2k.log',
    #     },
    #
    # 'Thunderbird': {
    #     'log_file': 'Thunderbird/Thunderbird_2k.log',
    #     },
    #
    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        },
    #
    # 'Linux': {
    #     'log_file': 'Linux/Linux_2k.log',
    #     },
    #
    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod_2k.log',
    #     },
    #
    # 'HealthApp': {
    #     'log_file': 'HealthApp/HealthApp_2k.log',
    #     },
    #
    # 'Apache': {
    #     'log_file': 'Apache/Apache_2k.log',
    #     },
    #
    # 'Proxifier': {
    #     'log_file': 'Proxifier/Proxifier_2k.log',
    #     },
    #
    # 'OpenSSH': {
    #     'log_file': 'OpenSSH/OpenSSH_2k.log',
    #     },
    #
    # 'OpenStack': {
    #     'log_file': 'OpenStack/OpenStack_2k.log',
    #     },
    #
    # 'Mac': {
    #     'log_file': 'Mac/Mac_2k.log',
    #     },
}


input = 'data/'
print("=== LogisticRegression ===")
for dataset, setting in settings.items():
    print("=== working on ", dataset, " ===")
    score = 0.0
    t = 0.0
    for iter in range(0, 10):
        print("running on test", iter)
        input_dir = os.path.join(input, dataset)
        start = time.time()
        train_wordvector = WordVector(indir = input_dir, logname = dataset, step = 1)
        train_x, train_y = train_wordvector.run()

        model = LogisticRegression()
        model.fit(train_x, train_y)
        end = time.time()

        test_wordvector = WordVector(indir = input_dir, logname = dataset, step = 2)
        test_x,test_y = test_wordvector.run()
        pred_y = model.predict(test_x)
        score = score + metrics.accuracy_score(test_y, pred_y)
        t = t + end-start

    print('评估:')
    print('准确率： ', score/10)
    print('算法消耗时间为： ', t/10,' 秒')
    print()

print()