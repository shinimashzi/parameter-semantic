from gensim.models import word2vec
import pandas as pd
import numpy as np
import random
import ast
import os


# prepare for model
# prepare input x, output y

class WordVector:
    def __init__(self, indir='/', outdir='/', logname="", step=0):
        self.indir = indir
        self.savepath = outdir
        self.df_log = None
        self.logname = logname
        self.step = step
        self.template_semantic = None

# 返回 x， y
    def run(self):
        self.load()
        self.preprocess()
        return self.word2vec()

# 取id， event， content， eventid， ParameterList
# train随机取1600行， test取剩下的400行
    def load(self):
        input_dir = os.path.join(self.indir, self.logname + "_2k.log_structured.csv")
        df = pd.read_csv(input_dir)
        save_dir = os.path.join(self.indir, 'test_index.txt')

        if self.step == 1:
            print("=== loading train data ", self.logname, " ===")
            seq = [i for i in range(0,2000)]
            train_index = random.sample(seq, 1600)
            train_index = list(np.sort(train_index))
            test_index = [x for x in range(0, 2000) if x not in train_index]
            message = open(save_dir, 'w')
            print(test_index, file = message)
            message.close()
            data=pd.DataFrame(df, index=train_index)
        else:
            save_dir = os.path.join(self.indir, 'test_index.txt')
            test_index = open(save_dir, 'r')
            test_index = test_index.read()
            test_index = ast.literal_eval(test_index)
            data = pd.DataFrame(df, index=test_index)
        data.index = range(len(data))
        self.df_log = data
        template_semantic_dir = os.path.join(self.indir, 'Template-Semantic.csv')
        self.template_semantic = pd.read_csv(template_semantic_dir)

# 由于直接用Content得出vector可能会有一些参数由于与上下文无空格原因而无法得出，所以用的是Template内插入参数来生成词向量。
    def preprocess(self):
        events = self.df_log['EventTemplate']
        self.data = []
        allowed_punctuation = ['<', '*', '>', ',', '.',' ']
        for idx, event in enumerate(events):
            d = ""
            for idz, s in enumerate(event):
                if s.isalnum() or s in allowed_punctuation:
                    d = d + s
                else:
                    d = d + ' '
            da = ""
            idy = 0
            vis = np.zeros(len(d), dtype=np.int)
            parameters = ast.literal_eval(self.df_log['ParameterList'][idx])
            num = len(parameters)
            for idz, s in enumerate(d):
                if vis[idz] == 1:
                    continue
                if d[idz:idz+3] == '<*>':
                    for x in range(idz, idz+3):
                        vis[x] = 1
                    if idy < num :
                        da = da + ' ' + parameters[idy] + ' '
                        idy = idy + 1
                else:
                    da = da + s
            self.data.append(da)

# 训练词向量,返回 input x, output y。
    def word2vec(self):
        modelpath = 'model/m2vmodel'
        formatpath = os.path.join('Word_embeddings/',self.logname)
        data = open('model/data.txt', 'w')
        str = ""
        for d in self.data:
            str = str + d + '. '
        print(str, file=data)
        data.close()

        sentences = word2vec.Text8Corpus("model/data.txt")
        if self.step == 1:
            model=word2vec.Word2Vec(sentences=sentences, sg=1, iter=10, min_count=1)
        else:
            model = word2vec.Word2Vec.load(modelpath)
            model.train(sentences, epochs=model.iter, total_examples=model.corpus_count)

        model.save(modelpath)
        model.wv.save_word2vec_format(formatpath, binary=False)

        parameterlist = []
        for parameters in self.df_log['ParameterList']:
            p = ast.literal_eval(parameters)
            parameterlist.append(p)

        wordvector_dict = {}
        for word in model.wv.index2word:
            wordvector_dict[word] = list(model[word])

        #df_out = pd.DataFrame(columns=["Position", "Parameter","WordVector"])

        df_out = []
        X = []
        Y = []
        for idx, paras in enumerate(parameterlist):
            idy = 0
            for idy, para in enumerate(paras):
                if para not in wordvector_dict:
                    continue
                df_out.append([idx, idy, self.df_log['EventId'][idx], para, wordvector_dict[para]])
                X.append(wordvector_dict[para])
                flag = 0
                for idz, r in self.template_semantic.iterrows():
                    if r['EventId'] == self.df_log['EventId'][idx] and r['Position'] == idy:
                            Y.append(r['Semantic'])
                            flag = 1
                            break
                if flag == 0:
                    print(idx, idy, paras, self.df_log['EventId'][idx])

        return X,Y