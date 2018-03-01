import numpy as np

class PerformanceMeasure():

    def __init__(self, real_list, pred_list):
        self.real = real_list
        self.pred = pred_list
        self.aee_value  = []
        self.fpa_value=0


    def AEE(self):
        '''
        求每一类模块上的平均绝对误差（average absolute error）
        real_list指测试集中每个模块的真实缺陷个数
        pred_list指训练出的回归模型对测试集中每个模块进行预测得出的预测值
        如real_list=[2,3,0,0,1,1,0,5,3]
         pred_list=[1,1,1,0,1,0,0,3,2]
         输出结果就为0:0.33, 1:0.5,  2:1,  3:1.5,  5:2
        '''
        only_r = np.array(list(set(self.real)))
        #only_r=[0,1,2,3,5]

        for i in only_r:
            r_index = np.where(self.real == i)
        #i=0的时候，r_index=【2，3】

            sum=0
            for k in r_index:
                sum=sum+abs((self.real[k]-self.pred[k]).sum())
                devi=sum*1.0/len(k)
                self.aee_value.append(devi)

        return only_r, self.aee_value



    def FPA(self):
        '''
        有四个模块m1,m2,m3,m4，真实缺陷个数分别为1，4，2，1,self.real=[1，4，2，1]
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,self.pred=[0,3,5,1]
        预测出的排序为m3>m2>m4>m1
        fpa=1/4 *1/8 *(4*2+3*4+2*1+1*1)=0.718
        '''
        K = len(self.real)
        N = np.sum(self.real)
        sort_axis = np.argsort(self.pred)
        testBug = np.array(self.real)
        testBug = testBug[sort_axis]
        P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
        return P
