import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def yee_1d(M, N, t0, fields=None, boundary='pmc'):
    # imp0 = 377.0

    if fields is None:
        Ez = np.zeros(M + 1)
        By = np.zeros(M + 1)
    else:
        Ez, By = fields
        assert Ez.shape == (M + 1,) and By.shape == (M + 1,)

    if boundary == 'pmc':
        for t in range(1, N + 1):
            Ez[1:] = (By[1:] - By[:-1]) + Ez[1:]
            Ez[M//3] += np.exp(-((t + t0) - 30) ** 2 / 100)
            By[:-1] = (Ez[1:] - Ez[:-1]) + By[:-1]

    else:
        raise NotImplementedError

    return Ez, By


def yee_3D(M, N, t0, fields=None, boundary='pmc'):
    # imp0 = 377.0
    stability_constant = 1 / np.sqrt(3)

    I, J, K = M
    if fields is None:
        Ex = np.zeros((I + 1, J + 1, K + 1))
        Ey = np.zeros((I + 1, J + 1, K + 1))
        Ez = np.zeros((I + 1, J + 1, K + 1))
        Bx = np.zeros((I + 1, J + 1, K + 1))
        By = np.zeros((I + 1, J + 1, K + 1))
        Bz = np.zeros((I + 1, J + 1, K + 1))
    else:
        Ex, Ey, Ez, Bx, By, Bz = fields
        # assert Ez.shape == (M + 1,) and By.shape == (M + 1,)

    if boundary == 'pmc':
        for t in range(1, N + 1):
            print(t)
            for i in range(1, I):
                for j in range(1, J + 1):
                    for k in range(1, K):
                        if i != I:
                            Ex[i, j, k] = stability_constant * ((Bz[i, j, k] - Bz[i, j - 1, k]) - (By[i, j, k] - By[i, j, k - 1])) + Ex[i, j, k]
                        if j != J:
                            Ey[i, j, k] = stability_constant * ((Bx[i, j, k] - Bx[i, j, k - 1]) - (Bz[i, j, k] - Bz[i - 1, j, k])) + Ex[i, j, k]
                        if k != K:
                            Ex[i, j, k] = stability_constant * ((By[i, j, k] - By[i - 1, j, k]) - (Bx[i, j, k] - Bx[i, j - 1, k])) + Ex[i, j, k]

                        Ez[I // 3, J // 3, K // 3] += np.exp(-((t + t0) - 30) ** 2 / 100)

            for i in range(0, I):
                for j in range(0, J):
                    for k in range(0, K):
                        if i != 0:
                            Bx[i, j, k] = -stability_constant * ((Ez[i, j + 1, k] - Ez[i, j, k]) - (Ey[i, j, k + 1] - Ey[i, j, k])) + Bx[i, j, k]
                        if j != 0:
                            By[i, j, k] = -stability_constant * ((Ex[i, j, k + 1] - Ex[i, j, k]) - (Ez[i + 1, j, k] - Ez[i, j, k])) + Bx[i, j, k]
                        if k != 0:
                            Bx[i, j, k] = -stability_constant * ((Ey[i + 1, j, k] - Ey[i, j, k]) - (Ex[i, j + 1, k] - Ex[i, j, k])) + Bx[i, j, k]

    else:
        raise NotImplementedError

    return Ex, Ey, Ez, Bx, By, Bz


if __name__ == '__main__':
    M = 50
    N = 20
    Ex, Ey, Ez, _, _, _ = yee_3D((M, M, M), N, 0)

    x = np.arange(M + 1)
    X, Y = np.meshgrid(x, x)
    Z = Ex[:, :, M // 3]
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    plt.show()
