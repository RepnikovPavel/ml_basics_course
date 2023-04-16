import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pprint
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

dataset_base_path = 'C:/Users/User/Desktop/ready_algs_for_ds/hw5/dataset'
dataset_full_path = os.path.join(dataset_base_path, 'TRAIN.csv')

if __name__ == '__main__':

    table = pd.read_csv(dataset_full_path)
    table = table.drop(columns=['Unnamed: 0'])
    le_cut = LabelEncoder()
    le_cut.fit(table.cut.unique())
    le_cut_clusses = le_cut.classes_
    table['cut'] = table['cut'].replace(le_cut_clusses,
                                        le_cut.transform(le_cut_clusses))

    le_color = LabelEncoder()
    le_color.fit(table.color.unique())
    le_color_clusses = le_color.classes_
    table['color'] = table['color'].replace(le_color_clusses,
                                            le_color.transform(le_color_clusses))

    le_clarity = LabelEncoder()
    le_clarity.fit(table.clarity.unique())
    le_clarity_clusses = le_clarity.classes_
    table['clarity'] = table['clarity'].replace(le_clarity_clusses,
                                                le_clarity.transform(le_clarity_clusses))

    table = shuffle(table, random_state=42)
    X = table[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
    y = table[['price']]
    criterions = [
        {'criterion': 'squared_error', 'params': {'depth': 12}, 'quality': 0.0},
        {'criterion': 'friedman_mse', 'params': {'depth': 16}, 'quality': 0.0},
        {'criterion': 'poisson', 'params': {'depth': 22}, 'quality': 0.0},
        {'criterion': 'squared_error', 'params': {'depth': 45}, 'quality': 0.0},
        {'criterion': 'friedman_mse', 'params': {'depth': 95}, 'quality': 0.0},
        {'criterion': 'poisson', 'params': {'depth': 33}, 'quality': 0.0},

    ]
    np.random.seed(42)
    scores_list = []
    for criterion in criterions:
        reg = DecisionTreeRegressor(criterion=criterion['criterion'], max_depth=criterion['params']['depth'],
                                    random_state=42)
        scores = cross_val_score(reg, X, y, cv=10)
        # scoring='r2'
        criterion['quality'] = np.mean(scores)
        pprint.pprint(criterion)

    # pprint.pprint(criterions)
