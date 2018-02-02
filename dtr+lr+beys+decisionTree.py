from sklearn import linear_model

import numpy as np

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing

from RandomUnderSampler import RandomUnderSampler


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
    #rus 下采样
    rus_training_data_X, rus_training_data_y, id_ = RandomUnderSampler(ratio=1.0,return_indices=True).fit_sample(training_data_X, training_data_y)


    rus_pred = pred_result(rus_training_data_X, rus_training_data_y, testing_data_X)
    #smote上采样
    smote_training_data_X, smote_training_data_y = Processing().dealData(training_data_X, training_data_y,ratio=1.0)

    smote_pred = pred_result(smote_training_data_X, smote_training_data_y, testing_data_X)

    for i in y_pred:

        fpa = PerformanceMeasure(testing_data_y,i).FPA()

        key, val = PerformanceMeasure(testing_data_y,i).AEE()

        mad_dict = dict(zip(key,val))

        print('No anything FPA',fpa)

        print('No anything Meanad',mad_dict)

    for i in rus_pred:

        fpa = PerformanceMeasure(testing_data_y,i).FPA()

        key, val = PerformanceMeasure(testing_data_y,i).AEE()

        mad_dict = dict(zip(key,val))

        print('Rus FPA',fpa)

        print('Rus Meanad',mad_dict)

    for i in smote_pred:

        fpa = PerformanceMeasure(testing_data_y,i).FPA()

        key, val = PerformanceMeasure(testing_data_y,i).AEE()

        mad_dict = dict(zip(key,val))

        print('Smote FPA',fpa)

        print('Smote Meanad',mad_dict)

# 决策树  线性回归 贝叶斯

if __name__ == '__main__':
  
    bootstrap()











