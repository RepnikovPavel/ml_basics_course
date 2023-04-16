import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

N_ESTIMATORS = 25
MAX_DEPTH = 14
SUBSPACE_DIM = 2

# n_estimators=25, max_depth=15, subspaces_dim=2, random_state=42

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

class random_forest(object):
    subsample_idxs = []
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state
        self.trees = [DecisionTreeClassifier(max_depth=max_depth, random_state=random_state) for el in range(n_estimators)]
        """
          Задайте все необходимые поля в рамках конструктора класса
        """

    def fit(self, X, y):
        for i in range(self.n_estimators):
            s = sample(X,self.subspaces_dim)
            bootstrap_indices = s.bootstrap_sample(X)
            X_sampled, y_sampled = s.get_subsample(X, y, s.idx_subspace, bootstrap_indices)
            self.subsample_idxs.append(s.idx_subspace.tolist())
            self.trees[i].fit(X_sampled,y_sampled)

    def predict(self, X):
        n_of_objects = X.shape[0]
        predictions = np.zeros(shape=(n_of_objects,))
        for i in range(n_of_objects):
            object_ = X[i]
            answers = []
            for j in range(len(self.trees)):
                subsamplde_idx   = self.subsample_idxs[j]
                object_to_model  = object_[subsamplde_idx].reshape(1, -1)
                # print(object_to_model)
                pred = self.trees[j].predict(object_to_model)[0]
                answers.append(pred)
            counts = np.bincount(answers)
            mean_answer = np.argmax(counts)
            predictions[i] = mean_answer
        return predictions

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    # n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    rf = random_forest(n_estimators=25, max_depth=15, subspaces_dim=2, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    print(accuracy_score(y_true=y_test, y_pred=preds))
