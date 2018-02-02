from sklearn import linear_model

import numpy as np

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing

def pred_result(training_data_X, training_data_y,test_data_X):

    '''

    return: 3个回归模型对test_data_X的预测值，预测值会取整

    '''

    dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)

    lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)

    bayes = BayesianRidge().fit(training_data_X, training_data_y)

    return [np.around(dtr.predict(test_data_X)), np.around(lr.predict(test_data_X)), np.around(bayes.predict(test_data_X))]



def bootstrap():

    dataset = Processing().import_data()

    training_data_X, training_data_y, testing_data_X, testing_data_y= Processing().separate_data(dataset)

    y_pred = pred_result(training_data_X, training_data_y, testing_data_X)

    for i in y_pred:

        fpa = PerformanceMeasure(testing_data_y,i).FPA()

        key, val = PerformanceMeasure(testing_data_y,i).AEE()

        mad_dict = dict(zip(key,val))

        print('FPA',fpa)

        print('Meanad',mad_dict)



# 决策树  线性回归 贝叶斯

if __name__ == '__main__':

    bootstrap()











