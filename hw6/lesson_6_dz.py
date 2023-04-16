import numpy as np

np.random.seed(42)


class sample(object):

    def __init__(self, X, n_subspace):
        self.idx_subspace = self.random_subspace(X, n_subspace)

    def __call__(self, X, y):
        idx_obj = self.bootstrap_sample(X)
        X_sampled, y_sampled = self.get_subsample(X, y, self.idx_subspace, idx_obj)
        return X_sampled, y_sampled

    @staticmethod
    def bootstrap_sample(X, random_state=42):
        """
        Заполните тело этой функции таким образом, чтобы она возвращала массив индексов выбранных при помощи бэггинга индексов.
        Пользуйтесь только инструментами, реализованными в numpy.random, выставляя везде, где это необходимо, random_state=42
        """
        r = np.random.RandomState()
        x_shape = X.shape
        n_of_features = x_shape[1]
        n_of_objects = x_shape[0]
        all_objects_indexes = np.arange(start=0, stop=n_of_objects, dtype=int)
        selected_objects_indexes = r.choice(all_objects_indexes, size=n_of_objects, replace=True)
        unique_indexes = np.unique(selected_objects_indexes)
        return unique_indexes

    @staticmethod
    def random_subspace(X, n_subspace, random_state=42):
        """
        Заполните тело этой функции таким образом, чтобы она возвращала массив индексов выбранных при помощи метода случайных подпространств признаков
        Количество этих признаков передается при помощи аргумента n_subspace
        Пользуйтесь только инструментами, реализованными в numpy.random, выставляя везде, где это необходимо, random_state=42
        """
        # выбрать случайные столбцы. n уникальных случайно выбранных?
        r = np.random.RandomState()
        x_shape = X.shape
        n_of_features = x_shape[1]
        n_of_objects = x_shape[0]
        all_features_indexes = np.arange(start=0, stop=n_of_features, dtype = int)
        selected_features = r.choice(all_features_indexes, size=n_subspace, replace=False)

        return selected_features

    @staticmethod
    def get_subsample(X, y, idx_subspace, idx_obj):
        """
        Заполните тело этой функции таким образом, чтобы она возвращала подвыборку x_sampled, y_sampled
        по значениям индексов признаков(idx_subspace) и объектов(idx_obj) , которые должны в неё попасть
        """
        n_features = idx_subspace.shape[0]
        n_objects = idx_obj.shape[0]
        X_sampled = np.zeros(shape=(n_objects,n_features))
        y_sampled = np.zeros(shape=(n_objects,))
        for i in range(n_objects):
            object_index = idx_obj[i]
            for j in range(n_features):
                feature_index = idx_subspace[j]
                X_sampled[i][j] = X[object_index][feature_index]
            y_sampled[i] = y[object_index]

        return X_sampled,y_sampled




if __name__ == '__main__':
    X = np.array([[1.1, 2.2, 3], [4, 5, 6], [7, 8, 9]])
    Y = np.array([1, 2, 3])
    s = sample(X, 2)

    bootstrap_indices = s.bootstrap_sample(X)
    X_sampled, y_sampled = s.get_subsample(X, Y, s.idx_subspace, bootstrap_indices)

    print(bootstrap_indices)
    print(s.idx_subspace)
    print(X_sampled)
    print(y_sampled)
