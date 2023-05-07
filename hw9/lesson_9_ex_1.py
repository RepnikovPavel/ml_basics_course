import os

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme()

if __name__ =='__main__':
    project_path=  'C:/Users/User/PycharmProjects/kurs_5_sem_2_mfk_ml'
    data_base_path = os.path.join(project_path, './lesson_9/data')
    filepaths = [os.path.join(data_base_path,path_) for path_ in os.listdir(data_base_path)]
    for path_ in filepaths:
        data_ = pd.read_csv(path_)
        x_ = data_.iloc[:,0]
        y_  = data_.iloc[:,1]
        labels_ = data_.iloc[:,2]
        number_of_clusters = np.unique(labels_)

        plt.figure(figsize=(12, 12))
        plt.scatter(x_, y_,c=labels_)
        plt.title(path_+' nuber of cl={}'.format(len(number_of_clusters)))
        plt.show()