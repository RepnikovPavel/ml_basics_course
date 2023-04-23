from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as auc

"""
TODO: make additional imports here
"""


X, y = load_breast_cancer(return_X_y=True)
X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

"""
In the following part of code specify algorithms with their own parameters by yourself
"""
tree = DecisionTreeClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True)

"""
TODO: fit estimators and find best one
"""
scaler = StandardScaler()
scaler.fit(X_train)
transormed_x_train = scaler.transform(X_train)
tree.fit(X_train,y_train)
lr.fit(X_train,y_train)
knn.fit(X_train,y_train)
svm.fit(X_train,y_train)


transformed_x_test = scaler.transform(x_test)
tr_pred = tree.predict(x_test)
lr_pred= lr.predict(x_test)
knn_pred = knn.predict(x_test)
svm_pred = svm.predict(x_test)


print('tree {}'.format(roc_auc_score(y_test,tr_pred)))
print('lr {}'.format(roc_auc_score(y_test,lr_pred)))
print('knn {}'.format(roc_auc_score(y_test,knn_pred)))
print('svm {}'.format(roc_auc_score(y_test,svm_pred)))

print('tree {}'.format(auc(y_test,tr_pred)))
print('lr {}'.format(auc(y_test,lr_pred)))
print('knn {}'.format(auc(y_test,knn_pred)))
print('svm {}'.format(auc(y_test,svm_pred)))