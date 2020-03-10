# encoding: utf-8
from Wordvector import WordVector
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
import pandas as pd
import time
import os

settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        },
    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        },

}

input = 'data/'

df_data = []
print("=== RandomForest === ")
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

        model = RandomForestClassifier()
        model.fit(train_x, train_y)
        end = time.time()

        test_wordvector = WordVector(indir = input_dir, logname = dataset, step = 2)
        test_x,test_y = test_wordvector.run()
        pred_y = model.predict(test_x)
        score = score + metrics.accuracy_score(test_y, pred_y)
        t = t + end-start

    score = score / 10
    t = t / 10
    print('评估:')
    print('准确率： ', score)
    print('算法消耗时间为： ', t, ' 秒')
    print()
    df_data.append([dataset, score, t])
df = pd.DataFrame(df_data, columns=['Log', 'Accuracy', 'Time'])
outdir = 'result/RandomForest.csv'
df.to_csv(outdir)
print()