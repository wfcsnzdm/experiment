import numpy as np

class MeanAbsoluteDeviation():
    def __init__(self, real_list, pred_list):
        self.real = real_list
        self.pred = pred_list
        self.rs  = []

    def Absolute(self):
        self.result = abs(self.real - self.pred)
        return self.Deviation()

    def Deviation(self):
        only_r = np.array(list(set(self.real)))
        for i in only_r:
            r_index = np.where(self.real == i)
            for k in r_index:
                self.count = len(k)
                devi = self.result[k].sum()/self.count
            self.rs.append(devi)
        return only_r, self.rs

