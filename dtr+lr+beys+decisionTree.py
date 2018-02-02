from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
import os

from MAD import MeanAbsoluteDeviation


def ImportData(folderName="data"):
    '''

    :param folderName:文件夹的名字
    :return: 所有文件下的数据
    '''
    dataSet = pd.core.frame.DataFrame()
    folderPath = folderName + '//'  # In Mac the path use '/' to identify the secondary path
    for root, dirs, files in os.walk(folderPath):
        for file in files:
            filePath = os.path.join(root, file)
            data1 = pd.read_csv(filePath)
            dataSet = dataSet.append(data1, ignore_index=True)
    return dataSet

def SeparateData(OriginalData):
    '''

    :param OriginalData:整个数据集
    :return: 划分好的 训练集和测试集
    '''
    OriginalData = OriginalData.iloc[:,3:]
    OriginalData = np.array(OriginalData)
    TrainingData = resample(OriginalData)
    k = len(TrainingData[0])
    #先转换成list 在进行数据筛选
    OriginalData = OriginalData.tolist()
    TrainingData = TrainingData.tolist()
    TestingData =[]
    for i in OriginalData:
        if i not in TrainingData:
            TestingData.append(i)
    TestingData = np.array(TestingData)
    TrainingData = np.array(TrainingData)
    TrainingDataX = TrainingData[:,0:k-2]
    TrainingDatay = TrainingData[:,k-1]
    TestingDataX  = TestingData[:,0:k-2]
    TestingDatay = TestingData[:,k-1]
    print(len(TestingDataX)/len(TrainingDataX))
    print()
    return  TrainingDataX, TrainingDatay,TestingDataX,TestingDatay

def FPA1(testBug, testPre):
    '''

    :param testBug: 真实的bug
    :param testPre: 预测的bug
    :return: fpa值
    '''
    K = len(testBug)
    N = np.sum(testBug)
    sort_axis = np.argsort(testPre)
    testBug = np.array(testBug)
    testBug = testBug[sort_axis]
    P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
    return P

def ResultOfNormal(X, y,X_test):
    '''

    :param X:训练集X
    :param y: 训练集y
    :param X_test: 测试集X
    :return: 3个模型的预测数据在一个list里面分别是决策树，线性回归，贝叶斯回归
    '''
    dtr = DecisionTreeRegressor().fit(X, y)
    lr = linear_model.LinearRegression().fit(X, y)
    beys = BayesianRidge().fit(X, y)
    return [dtr.predict(X_test).astype(int), lr.predict(X_test).astype(int), beys.predict(X_test).astype(int)]

def bootstap():
    dataSet = ImportData()
    X_train, y_train, X_test,y_test= SeparateData(dataSet)
    y_pred = ResultOfNormal(X_train,y_train,X_test)
    for i in y_pred:
        fpa = FPA1(y_test,i)
        key, val = MeanAbsoluteDeviation(y_test,i).Absolute()
        mad_dict = dict(zip(key,val))
        print('FPA',fpa)
        print('Meanad',mad_dict)
# 决策树  线性回归 贝叶斯
if __name__ == '__main__':
    bootstap()





