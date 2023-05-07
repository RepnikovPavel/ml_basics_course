import copy
import os

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import seaborn as sns
import pandas as pd

sns.set_theme()

import numpy as np
class KMeans(object):
    K: int
    cms_: np.array
    def __init__(self, K, init):
        """
        :param K: possible number of clusters
        :param init: start approx for center of masses
        """
        self.K = K
        # TODO: write your code here
        self.cms_ = init

    def __np_dist(self, a: np.array, b: np.array):
        return np.linalg.norm(a-b)

    def __is_end(self,old_cms_:np.array, new_cms_:np.array)->bool:
        dits_ = [self.__np_dist(old_cms_[i], new_cms_[i]) for i in range(self.K)]
        max_dits_ = np.max(dits_)
        if max_dits_ < 0.001:
            return True
        else:
            return False

    def fit(self, X: np.array):
        """

        :param X: array of vectors with features
        :return:None
        """
        # TODO: write your code here

        while True:
            current_cms_ = self.cms_
            shape_1=  len(self.cms_)
            shape_2 = len(self.cms_[0])
            new_cms = np.zeros(shape=(shape_1,shape_2))
            cls_labels = self.predict(X)
            cls_ = [[] for i in range(self.K)]
            for i in range(len(X)):
                cls_[cls_labels[i]].append(X[i])

            for j in range(self.K):
                cm_ = np.zeros(shape=len(X[0]))
                for k in range(len(cls_[j])):
                    cm_ += cls_[j][k]
                cm_ = cm_/len(cls_[j])
                new_cms[j] = cm_
            if self.__is_end(old_cms_=current_cms_, new_cms_=new_cms):
                break
            else:
                self.cms_ = new_cms


    def predict(self, X: np.array):
        """

        :param X: array of vectors with features
        :return: labels of input objects
        """
        N = len(X)
        K = self.K
        labels_ = np.zeros(shape=(N,), dtype=np.intc)
        dists_buffer_ = np.zeros(shape=(K,))
        for i in range(len(X)):
            x_ = X[i]
            for j in range(K):
                dists_buffer_[j] = self.__np_dist(x_, self.cms_[j])
            labels_[i] = np.argmin(dists_buffer_)
        return labels_

if __name__ =='__main__':
    project_path = 'C:/Users/User/PycharmProjects/kurs_5_sem_2_mfk_ml'
    data_path = os.path.join(project_path, './lesson_9/data/2.csv')
    data_ = pd.read_csv(data_path)
    x_ = data_.iloc[:, 0]
    y_ = data_.iloc[:, 1]
    labels_ = data_.iloc[:, 2]
    number_of_clusters = np.unique(labels_)

    cms_ = np.asarray([
        [0.0, 40.0],
        [50.0, 0.0],
        [80.0,70.0],
        [90.0, 50.0]
    ])
    clf_ = KMeans(K=4, init=cms_)
    X = np.transpose(np.asarray([x_, y_]))
    clf_.fit(X = X)
    predict_ = clf_.predict(X)


    plt.figure(figsize=(12, 12))
    plt.scatter(x_, y_, c=predict_)
    plt.title(data_path + ' nuber of cl={}'.format(len(number_of_clusters)))
    plt.show()