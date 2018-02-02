import numpy as np
import pandas as pd
from sklearn.utils import resample
import os
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
