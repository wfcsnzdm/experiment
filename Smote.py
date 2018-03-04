import random

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import os
class Smote:
    # samples的最后一列是类标，都是1
    def __init__(self, samples, Y, N=10, k=5):
        self.n_samples = len(samples)
        self.Y = Y
        self.n_attrs = len(samples[0])
        self.N = N
        self.k = k + 1
        self.samples = samples

    def over_sampling(self):
        if self.N < 100:
            old_n_samples = self.n_samples
            print("old_n_samples", old_n_samples)

            self.n_samples = int(float(self.N) / 100 * old_n_samples)
            print("n_samples", self.n_samples)

            keeps = np.random.permutation(old_n_samples)[:self.n_samples]
            print("keep", keeps)

            new_samples = [self.samples[keep] for keep in keeps]
            print("new_samples", new_samples)

            self.samples = new_samples
            print("self.samples", self.samples)

            self.N = 100

        N = int(self.N / 100)  # 每个少数类样本应该合成的新样本个数
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        self.label = []
        self.new_index = 0
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)

        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples, return_distance=False)[0]
            # 存储k个近邻的下标
            self.__populate(N, i, nnarray)
        return self.synthetic, self.label

    # 从k个邻居中随机选取N次，生成N个合成的样本
    def __populate(self, N, i, nnarray):
        for n in range(N):
            nn = np.random.randint(1, self.k)
            dif = self.samples[nnarray[nn]] - self.samples[i]  # 包含类标
            gap = np.random.rand(1, self.n_attrs)
            self.synthetic[self.new_index] = self.samples[i] + gap.flatten() * dif
            dist2 = (float)(np.linalg.norm(self.synthetic[self.new_index] - self.samples[nnarray[nn]]))
            dist1 = (float)(np.linalg.norm(self.synthetic[self.new_index] - self.samples[i]))
            if (dist1 + dist2 != 0):
                self.label.append((dist1 * self.Y[nnarray[nn]] + dist2 * self.Y[i]) / (dist1 + dist2))
            else:
                self.label.append(self.Y[i])
            self.new_index += 1

