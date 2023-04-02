import sklearn
import numpy as np
np.random.seed(42)

class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_ = None


    def fit(self, x: np.array, y: np.array):
        # input - all train data
        ones = np.ones(shape=(x.shape[0],1))
        x_ = np.concatenate((x, ones), axis=1)
        self.coef_ = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_),x_)),np.transpose(x_)),y)
        # matmul
        # transpose
        # linalg.inv

    def predict(self, x: np.array):
        # input - full test data
        # tmp_1=  self.coef_[:-1]
        return np.dot(x, self.coef_[:-1])+self.coef_[-1]

def r2(y_true, y_pred):
    import numpy as np
    N = y_true.shape[0]
    De = 1/N*np.sum(np.square(y_pred-y_true))
    Dz = 1/N*np.sum(np.square(y_true-np.mean(y_true)))
    return 1-De/Dz


R_list = []
for i in range(1,6):
    filename = './{}.npy'.format(i)
    x = np.load(filename)
    lr = LinearRegression()
    y_true = np.expand_dims(x[:,1],axis=1)
    x_train = np.expand_dims(x[:,0],axis=1)
    lr.fit(x_train, y_true)
    y_pred = lr.predict(x_train)
    r_s= r2(y_true,y_pred)
    R_list.append(r_s)


print(R_list)
print(np.argmax(R_list))
print(np.argmin(R_list))