

# JUST RUN THIS CELL
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
np.random.seed(42)


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784')

    X = mnist.data.to_numpy()
    y = mnist.target.to_numpy()


    X = X[:2000]
    y = y[:2000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    SS = StandardScaler()
    SS.fit(X_train)
    X_train = SS.transform(X_train)


    N_COMPONENTS = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60]
    values = []
    for n_comp in N_COMPONENTS:
        pca = PCA(n_components=n_comp)
        pca.fit(X_train)
        clf = LogisticRegression(random_state=42)
        clf.fit(pca.transform(X_train), y_train)
        preds = clf.predict(pca.transform(SS.transform(X_test)))
        acc = accuracy_score(y_true=y_test, y_pred=preds)
        values.append(acc)
        print('n_comp {} acc {}'.format(n_comp, acc))
    print(N_COMPONENTS)
    print(values)
    argm = np.argmax(values)
    print(N_COMPONENTS[argm])
    print(values[argm])

    # plt.figure(figsize=(20,4))
    # for index, (image, label) in enumerate(zip(X[0:5], y[0:5])):
    #     plt.subplot(1, 5, index + 1)
    #     plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    #     plt.title('Training: %s\n' % label, fontsize = 20)

    plt.show()