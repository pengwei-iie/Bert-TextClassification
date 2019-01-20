# encoding:utf-8
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from doc_textLoad import *
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from service.client import BertClient

classfier = 'LR'
feature = 'Tfidf'
folds = 6
def main():
    # 汽车观点提取,分过词的数据
    # df_train = pd.read_csv('./data/ccf_car_train.csv')
    # df_test = pd.read_csv('./data/ccf_car_test.csv')
    # train = df_train['word_seg']
    # test_x = df_test['content']
    # target = df_train['class']
    # train = np.array(train)
    # target = np.array(target)
    # list = []
    # for i in target:
    #     list.append(i)
    bc = BertClient()
    train = np.load('./data/train_x.npy')
    train = list(train)
    train = bc.encode(train)
    target = np.load('./data/train_y.npy')
    test_x = np.load('./data/test_x.npy')
    test_y = np.load('./data/test_y.npy')

    test_x = list(test_x)
    test_x = bc.encode(test_x)
    # 分为训练集和测试的时候加上
    # train, test_x, target, test_y = train_test_split(train, target, test_size = 0.15, random_state = 0)
    # np.savetxt('./data/train_x', train_X, fmt='%s')
    # np.savetxt('./data/train_y', train_y, fmt='%s')
    # np.savetxt('./data/test_X', test_X, fmt='%s')
    # np.savetxt('./data/test_y', test_y, fmt='%s')
    # df_test = pd.read_csv(test_file)
    # df_train = df_train.drop(['article'], axis=1)
    # df_test = df_test.drop(['article'], axis=1)
    # ngram_range：tuple (min_n, max_n) 要提取的不同n-gram的n值范围的下边界和上边界。 将使用n的所有值，使得min_n <= n <= max_n。
    # max_features: 构建一个词汇表，该词汇表仅考虑语料库中按术语频率排序的最高max_features
    # if feature == 'Count':
    #     vectoriser = CountVectorizer(ngram_range=(1, 2), min_df = 3)
    # elif feature == 'Tfidf':
    #     vectoriser = TfidfVectorizer(ngram_range=(1, 5), min_df = 3, max_df = 0.7)
    # # 构建特征，先训练
    # vectoriser.fit(train)
    # 训练完进行归一化  总共有315503个词，过滤小于3的，剩下23546, max_df 貌似没用
    # (7957, 2082), (1990, 2082) type:crs_matrix
    # train_X = vectoriser.transform(train)
    # test_X = vectoriser.transform(test_x)
    # y_train = df_train['class'] - 1
    # train_X = np.array(train_X.data).reshape(train_X.shape)

    # 开始构建分类器
    if classfier == 'LR':
        '''
        c：正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化
        '''
        rg = LogisticRegression(C=1)
        rg.fit(train, target)
        y_pred = rg.predict(test_x)
    # elif classfier == 'NB':
    #     # 使用默认的配置对分类器进行初始化。
    #     mnb_count = MultinomialNB(alpha=0.2)
    #     # 使用朴素贝叶斯分类器，对CountVectorizer（不去除停用词）后的训练样本进行参数学习。
    #     mnb_count.fit(train_X, target)
    #     y_pred = mnb_count.predict(test_X)
    #
    # elif classfier =='tree':
    #     DT = tree.DecisionTreeClassifier()
    #     DT.fit(train_X, target)
    #     y_pred = DT.predict(test_X)
        '''
            kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。

        　　 kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
        '''
    # C=0.1 准确率高召回率低   C = 0.8
    # elif classfier =='RT':
    #     sv = RandomForestClassifier(n_estimators=400)
    #     sv.fit(train_X, target)
    #     # y_hat = sv.predict(train_X)
    #     y_pred = sv.predict(test_X)
    #     scores = cross_val_score(sv, train_X, target, cv=5)
    #     print(scores)
    # 从sklearn.metrics 导入 classification_report。

    # 输出更加详细的其他评价分类性能的指标。
    print('classifier is : ' + classfier + '\tFeature is : ' + feature)
    print(classification_report(test_y, y_pred))
    # print(classification_report(target, y_hat))

    # test
    # df_test['subject'] = y_pred
    # df_result = df_test.loc[:, ['content_id', 'subject']]
    # print('last')
    # df_result.to_csv('./data/result.csv', index=False)

    # Second apporache
    # dataLoad = TextLoader('./data/', batch_size=128)
    # dataLoad_test = Loader_test('./data/', batch_size=1)
    # # 普通统计CountVectorizer提取特征向量
    # vectoriser = CountVectorizer(ngram_range=(1, 2), max_df=.9, min_df=3, max_features=10000)
    # rg = LogisticRegression(C=4, dual=True)
    #
    # # 训练得到文本的特征表示
    # filename = './data/vectoriser.sav'
    # if not os.path.exists(filename):
    #     for i in range(dataLoad.num_batches):
    #         x, y = dataLoad.next_batch()
    #         vectoriser.fit(x)
    #     pickle.dump(vectoriser, open(filename, 'wb'))
    # else:
    #     with open(filename, 'rb') as f:
    #         vectoriser = pickle.load(f)
    #
    # # # 训练得到逻辑回归模型
    # model_name = './data/model.sav'
    # if not os.path.exists(model_name):
    #     for i in range(dataLoad.num_batches):
    #         dataLoad.pointer = 0
    #         x, y = dataLoad.next_batch()
    #         # 训练完进行归一化
    #         x_train = vectoriser.transform(x)
    #         y_train = y - 1
    #         # 开始构建分类器
    #         rg.fit(x_train, y_train)
    #     pickle.dump(rg, open(model_name, 'wb'))
    # else:
    #     with open(model_name, 'rb') as f:
    #         rg = pickle.load(f)
    # # test
    # x_test = vectoriser.transform(dataLoad_test.df_test['word_seg'])
    # y_test = rg.predict(x_test)
    # dataLoad_test.df_test['class'] = y_test.to_list()
    # dataLoad_test.df_test['class'] += 1
    # df_result = dataLoad_test.df_test.loc[:, ['id', 'class']]
    # # print('last')
    # df_result.to_csv('./data/result.csv', index=False)
    # for i in range(dataLoad_test.num_batches):
    #     x = dataLoad_test.next_batch()
    #     x_test = vectoriser.transform(x)
    #     y_test = rg.predict(x_test)
    #     dataLoad_test.df_test['class'] = y_test.to_list()
    #     dataLoad_test.df_test['class'] += 1
    #     df_result = dataLoad_test.df_test.loc[:, ['id', 'class']]
    # df_result.to_csv('./data/result.csv', index=False)

    # Third
    # start = time.time()
    # file_path = './data/test_set.csv'  # 要拆分文件的位置
    # reader = pd.read_csv(file_path, chunksize=20000)
    # count = 0
    # for chunk in reader:
    #     print('save test_set%s.csv' % count)
    #     chunk.to_csv('test_set' + str(count) + '.csv', index=0)
    #     use = time.time() - start
    #     print('{:.0f}m {:.0f}s ...'.format(use // 60, use % 60))
    #     count += 1
    # 读取大文件，上面使用
    # df_train = read_files(6, 'train_set')
    # df_test = read_files(6, 'test_set')
    # df_train.drop(columns=['article', 'id'], inplace=True)
    # df_test.drop(columns=['article'], inplace=True)
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
    # vectorizer.fit(df_train['word_seg'])
    # x_train = vectorizer.transform(df_train['word_seg'])
    # x_test = vectorizer.transform(df_test['word_seg'])
    # y_train = df_train['class'] - 1

if __name__ == '__main__':
    main()
