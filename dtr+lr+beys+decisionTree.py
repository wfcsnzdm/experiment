

from sklearn import linear_model

import numpy as np

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing

from RandomUnderSampler import RandomUnderSampler

from RusAdaBoostRegressor import RAdaBoostRegressor

from SmoteAdaBoostRegressor import SAdaBoostRegressor

def pred_result(training_data_X, training_data_y,test_data_X):

    '''

    return: 3个回归模型对test_data_X的预测值，预测值会取整

    '''

    dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)

    lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)

    bayes = BayesianRidge().fit(training_data_X, training_data_y)

    return [np.around(dtr.predict(test_data_X)), np.around(lr.predict(test_data_X)), np.around(bayes.predict(test_data_X))]

def pred_result_boost(training_data_X, training_data_y, test_data_X, ratio =1, n_estimators =100):
    '''

    :return: 用rus下采样的adaboostRegressor，使用3个不同的回归模型预测，预测值取整。

    '''
    rng = np.random.RandomState(1)
    dtr = RAdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=n_estimators, random_state=rng).fit(training_data_X, training_data_y)

    lr = RAdaBoostRegressor(linear_model.LinearRegression(),n_estimators=n_estimators, random_state=rng).fit(training_data_X, training_data_y)

    bayes = RAdaBoostRegressor(BayesianRidge(),n_estimators = n_estimators, random_state=rng).fit(training_data_X, training_data_y)

    return [np.around(dtr.predict(test_data_X)), np.around(lr.predict(test_data_X)), np.around(bayes.predict(test_data_X))]

def pred_result_smoteboost(training_data_X, training_data_y, test_data_X, ratio =1, n_estimators =100):
    '''

    :return: 用smote上采样的adaboostRegressor，使用3个不同的回归模型预测，预测值取整。

    '''
    rng = np.random.RandomState(1)
    dtr = SAdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=n_estimators, random_state=rng).fit(training_data_X, training_data_y,ratio=ratio)

    lr = SAdaBoostRegressor(linear_model.LinearRegression(),n_estimators=n_estimators, random_state=rng).fit(training_data_X, training_data_y,ratio=ratio)

    bayes = SAdaBoostRegressor(BayesianRidge(),n_estimators = n_estimators, random_state=rng).fit(training_data_X, training_data_y,ratio=ratio)

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

    #rus + adaboostr2

    rus_boostr2_pred = pred_result_boost(training_data_X, training_data_y, testing_data_X, ratio=1.0)

    #smote + adaboostr2

    smote_boostr2_pred = pred_result_smoteboost(training_data_X, training_data_y, testing_data_X,ratio=1.0)


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

    for i in rus_boostr2_pred:

        fpa = PerformanceMeasure(testing_data_y,i).FPA()

        key, val = PerformanceMeasure(testing_data_y,i).AEE()

        mad_dict = dict(zip(key,val))

        print('rus_boostr2 FPA',fpa)

        print('rus_boostr2 Meanad',mad_dict)

    for i in smote_boostr2_pred:

        fpa = PerformanceMeasure(testing_data_y,i).FPA()

        key, val = PerformanceMeasure(testing_data_y,i).AEE()

        mad_dict = dict(zip(key,val))

        print('smote_boostr2 FPA',fpa)

        print('smote_boostr2 Meanad',mad_dict)
# 决策树  线性回归 贝叶斯

if __name__ == '__main__':

    bootstrap()

