import numpy as np
import pandas as pd
from sklearn.utils import resample
import os
from Smote import Smote
class Processing():
    def __init__(self):
        self.folder_name = "data"
    def import_data(self):

        '''

        读取文件夹中所有文件数据

        folder_name:文件夹的名字

        return: 文件夹下所有文件的数据

        '''

        dataset = pd.core.frame.DataFrame()

        folder_path = self.folder_name + '//'  # In Mac the path use '/' to identify the secondary path

        for root, dirs, files in os.walk(folder_path):

            for file in files:
                file_path = os.path.join(root, file)

                data1 = pd.read_csv(file_path)

                dataset = dataset.append(data1, ignore_index=True)

        return dataset

    def separate_data(self,original_data):

        '''

        用out-of-sample bootstrap方法产生训练集和测试集

        OriginalData:整个数据集

        return: 划分好的 训练集和测试集

        '''

        original_data = original_data.iloc[:, 3:]

        original_data = np.array(original_data)

        training_data = resample(original_data)  # 从originaldata中有放回的抽样，size(trainingdata)==size(originaldata)

        k = len(training_data[0])

        # 先转换成list 在进行数据筛选

        original_data = original_data.tolist()

        training_data = training_data.tolist()

        testing_data = []

        for i in original_data:

            if i not in training_data:
                testing_data.append(i)

        testing_data = np.array(testing_data)

        training_data = np.array(training_data)

        training_data_X = training_data[:, 0:k - 2]

        training_data_y = training_data[:, k - 1]

        testing_data_X = testing_data[:, 0:k - 2]

        testing_data_y = testing_data[:, k - 1]

        return training_data_X, training_data_y, testing_data_X, testing_data_y


    def refreshData(self,dataX, dataY):
        '''
        这个函数用来处理 datay非0的数据，因为要对非0的值进行上采样，所以把非0的值找出来。
        :param dataX: 原始数据集的X
        :param dataY: 原始数据集的y
        :return: 为0的特征个数，和非0的dataX 和 datay
        '''
        bugDataX = []
        bugDataY = []
        count = 0
        dataY = np.matrix(dataY).T
        dataX = np.array(dataX)
        for i in range(len(dataY)):
            if dataY[i] == 0:
                count += 1
            else:
                bugDataX.append(dataX[i])
                bugDataY.append(int(dataY[i]))
        return count, bugDataX, bugDataY

    def dealData(self,X, y,ratio=1):
        # SMOTE algorithm
        count, BugdataX, Bugdatay = self.refreshData(X, y)
        count = len(y) - count
        T = ratio * (len(y)-count) - count
        N = int(T/(count*0.01))
        add_X, add_y = Smote(BugdataX, Bugdatay, N, 5).over_sampling()
        # Conversion format
        add_y = np.array(add_y)
        y = np.hstack((y, add_y))
        X = np.vstack((X, add_X))
        return X, y
