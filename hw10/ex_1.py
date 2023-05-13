import numpy as np
from numpy.linalg import svd
from sklearn.neighbors import NearestNeighbors

class similaryty_analizer(object):
    def __init__(self, R: np.array):
        self.R = R
        self.n_users = len(R)
        self.n_items = len(R[0])

    def _get_svd(self, new_dim: int):
        # Выполните SVD-разложение
        if self.n_items==0 or self.n_users==0:
            return np.array([]),np.array([])
        R = self.R
        U, S, V = svd(R, full_matrices=True)

        # Выполните SVD-преобразование для снижения размерности
        U = U[:new_dim]
        S = S[:new_dim]
        V = V[new_dim:]

        # P = np.matmul(U, S)
        # Q = V

        P, Q = U @ np.diag(S), V.T

        return P, Q

    def get_similar_users(self, n_users: int, user_id: int):
        if self.n_items==0 or self.n_users==0:
            return np.asarray([])
        P, Q = self._get_svd(self.n_users)
        # P is users embeddings, Q is entities embeddings

        nn = NearestNeighbors(n_neighbors=n_users+1)
        nn = nn.fit(P)
        user_ = P[user_id, :]
        if len(user_.shape) == 1:
            user_ = user_.reshape(1, -1)
        neighbours = np.asarray(nn.kneighbors(user_, return_distance=False))[:, 1:].ravel()
        # [: 1,:].ravel()
        return neighbours


    def get_similar_items(self, n_items: int, item_id: int):
        if self.n_items==0 or self.n_users==0:
            return np.asarray([])
        P, Q = self._get_svd(self.n_items)

        nn = NearestNeighbors(n_neighbors=n_items+1)

        nn = nn.fit(Q)
        item_ = Q[item_id, :]
        if len(item_.shape) == 1:
            item_ = item_.reshape(1, -1)
        neighbours = nn.kneighbors(item_, return_distance=False)[:, 1:].ravel()
        return neighbours


if __name__ == '__main__':
    r1 = [1, 0, 0]
    r2 = [1, 1, 0]
    r3 = [1, 1, 1]

    R = np.array([r1, r2, r3])
    # R = np.array([[], [], []])
    SA = similaryty_analizer(R)
    print(SA.get_similar_users(1, 0))
    print(SA.get_similar_users(2, 0))
    print(SA.get_similar_users(2, 1))