import numpy as np


class friendadviser(object):
    def fit(self, R: np.array):
        self.R = R
        self.n_users = len(R)
        self.n_items = len(R[0])

    def _sim(self, u1: np.array, u2: np.array):
        # PMI
        # pmi_ = 0.0
        # for i in range(len(u1)):
        #     n_x_y   =       # число объектов, которые имели оба признака x и y
        #     n_x     =       # число объектов, которые имели признак x
        #     n_y     =       # число объектов, которые имели признак y
        #     pmi_ += n_x_y/(n_x*n_y)

        # u1r_ = np.sum(u1)/len(u1)
        # u2r_ = np.sum(u2)/len(u2)

        # binary & operator
        bitwise_and_ = u1*u2
        # n_x_y = np.sum(np.where(u1 == u2))
        n_x_y = np.sum(bitwise_and_)
        P_x_y = n_x_y/len(u1)
        P_y = np.sum(u2)/len(u2)
        P_x_cond_y = P_x_y/P_y
        P_x = np.sum(u1)/len(u1)
        sim_ = P_x_cond_y/P_x

        return sim_

    def U_idx(self, u0: np.array, alpha: float):
        list_ = []
        for i in range(self.n_users):
            other_ = self.R[i]
            if self._sim(u0, other_) >= alpha:
                list_.append(i)
        return np.asarray(list_)

    def find_friends(self, u0: np.array, how_many: int):
        list_ = np.zeros(shape=(self.n_users,))
        for i in range(self.n_users):
            list_[i] = self._sim(u0, self.R[i])
        indexes_ = np.argsort(list_)
        return np.flip(indexes_[-how_many:])


if __name__ =='__main__':
    u1 = np.array([1,0,1,0])
    u2 = np.array([0,0,1,0])
    R = [u1,u2]
    fv = friendadviser()
    fv.fit(R)
    print(fv._sim(u1,u2))