import pprint

import numpy as np
import matplotlib.pyplot as plt


def E(m, s_vec):
    m_sum = np.sum(s_vec[m+1:])
    all_sum = np.sum(s_vec)
    div = m_sum/all_sum
    if div<0.2:
        print(m)
    return div




if __name__ == '__main__':
    A = np.load('./lesson_8/PCA.npy')
    U, S, V = np.linalg.svd(A)
    pprint.pprint(S)
    E_m_vec = []
    m_vec=  []
    for i in range(1,len(S)+1):
        m_vec.append(i)
        E_m_vec.append(E(i,S))
    plt.plot(m_vec,E_m_vec)
    plt.show()

    print(1)