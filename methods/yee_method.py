import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def yee_1d(M, N, t0=0, t_end=None, x_end=None, initial_fields=None, boundary='pmc', pulse=False):
    '''
    :param initial_fields: Tuple of arrays ((Ez, By)) with shapes (M + 1,)
    '''

    if t_end is None and x_end is None:
        cp = 1
    else:
        c = 299792458
        if t_end is not None:
            k = t_end / N
        if x_end is not None:
            h = x_end / (M - 1)
        cp = c * k / h
        assert cp <= 1  # Courant number

    if initial_fields is None:
        Ez = np.zeros(M)
        By = np.zeros(M)
    else:
        Ez, By = initial_fields
        assert Ez.shape == By.shape == (M,)

    if boundary == 'pmc':
        for t in range(1, N + 1):
            Ez[1:] = cp * (By[1:] - By[:-1]) + Ez[1:]
            if pulse:
                Ez[M // 3] += np.exp(-((t + t0) - 30) ** 2 / 100)
            By[:-1] = cp * (Ez[1:] - Ez[:-1]) + By[:-1]
    else:
        raise NotImplementedError

    return Ez, By


def yee_3d(M, N, t0=0, initial_fields=None, boundary='pmc', pulse=False):
    '''
    :param initial_fields: Tuple of arrays ((Ex, Ey, Ez, Bx, By, Bz)) with shapes (M[0] + 1, M[1] + 1, M[2] + 1)
    '''

    # imp0 = 377.0
    stability_constant = 1 / np.sqrt(6)

    I, J, K = M
    if initial_fields is None:
        Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(I, J, K)
    else:
        Ex, Ey, Ez, Bx, By, Bz = initial_fields
        assert Ex.shape == (N + 1, I - 1, J, K) and \
               Ey.shape == (N + 1, I, J - 1, K) and \
               Ez.shape == (N + 1, I, J, K - 1) and \
               Bx.shape == (N + 1, I, J - 1, K - 1) and \
               By.shape == (N + 1, I - 1, J, K - 1) and \
               Bz.shape == (N + 1, I - 1, J - 1, K)

    if boundary == 'pmc':
        for t in range(1, N + 1):
            print(t)
            Ex[t, :, 1:-1, 1:-1] = Ex[t - 1, :, 1:-1, 1:-1] + stability_constant * \
                                                              ((Bz[t - 1, :, 1:, 1:-1] - Bz[t - 1, :, :-1, 1:-1]) -
                                                               (By[t - 1, :, 1:-1, 1:] - By[t - 1, :, 1:-1, :-1]))
            Ey[t, 1:-1, :, 1:-1] = Ey[t - 1, 1:-1, :, 1:-1] + stability_constant * \
                                                              ((Bx[t - 1, 1:-1, :, 1:] - Bx[t - 1, 1:-1, :, :-1]) -
                                                               (Bz[t - 1, 1:, :, 1:-1] - Bz[t - 1, :-1, :, 1:-1]))
            Ez[t, 1:-1, 1:-1, :] = Ez[t - 1, 1:-1, 1:-1, :] + stability_constant * \
                                                              ((By[t - 1, 1:, 1:-1, :] - By[t - 1, :-1, 1:-1, :]) -
                                                               (Bx[t - 1, 1:-1, 1:, :] - Bx[t - 1, 1:-1, :-1, :]))

            if pulse:
                Ez[t, 0, 1:-1, K // 2] = np.exp(-((t + t0) - 30) ** 2 / 100)

            Bx[t, 1:-1, :, :] = Bx[t - 1, 1:-1, :, :] - stability_constant * \
                                                        ((Ez[t, 1:-1, 1:, :] - Ez[t, 1:-1, :-1, :]) -
                                                         (Ey[t, 1:-1, :, 1:] - Ey[t, 1:-1, :, :-1]))
            By[t, :, 1:-1, :] = By[t - 1, :, 1:-1, :] - stability_constant * \
                                                        ((Ex[t, :, 1:-1, 1:] - Ex[t, :, 1:-1, :-1]) -
                                                         (Ez[t, 1:, 1:-1, :] - Ez[t, :-1, 1:-1, :]))
            Bz[t, :, :, 1:-1] = Bz[t - 1, :, :, 1:-1] - stability_constant * \
                                                        ((Ey[t, 1:, :, 1:-1] - Ey[t, :-1, :, 1:-1]) -
                                                         (Ex[t, :, 1:, 1:-1] - Ex[t, :, :-1, 1:-1]))

    else:
        raise NotImplementedError

    return Ex, Ey, Ez, Bx, By, Bz


def initialize_fields_3d(I, J, K):
    Ex = np.zeros((N + 1, I - 1, J, K))
    Ey = np.zeros((N + 1, I, J - 1, K))
    Ez = np.zeros((N + 1, I, J, K - 1))
    Bx = np.zeros((N + 1, I, J - 1, K - 1))
    By = np.zeros((N + 1, I - 1, J, K - 1))
    Bz = np.zeros((N + 1, I - 1, J - 1, K))
    return Ex, Ey, Ez, Bx, By, Bz


def initial_wave():
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, M, M)
    l = np.arange(M // 2)
    wave = np.sin(l / l[-1] * np.pi)
    start_y = M // 4
    stop_y = M // 4 + M // 2
    Ez[0, M // 2, start_y:stop_y, M // 2] = wave
    By[0, M // 2, start_y:stop_y, M // 2] = wave

    _, _, Ez, _, _, _ = yee_3d((M, M, M), N, initial_fields=(Ex, Ey, Ez, Bx, By, Bz))

    return Ex, Ey, Ez, Bx, By, Bz


def initial_analytical(periods=1):
    c = 299792458
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, M, M)
    Ez_plane_analytical = np.zeros((N, M, M))
    beta = periods / M

    sin_x = np.sin(2 * np.pi * np.linspace(0, M * beta, M)).reshape(M, 1, 1)
    sin_y = np.sin(2 * np.pi * np.linspace(0, M * beta, M)).reshape(1, M, 1)
    sin_z = np.sin(2 * np.pi * np.linspace(beta / 2, (M - 1 / 2) * beta, M - 1)).reshape(1, 1, M - 1)
    Ez[0, :, :, :] = sin_x * sin_y * sin_z

    cos_t = np.cos(np.sqrt(12) * np.pi * c * np.arange(N + 1)).reshape(N + 1, 1, 1)
    sin_x = sin_x.reshape(1, M, 1)
    sin_y = sin_y.reshape(1, 1, M)
    Ez_plane_analytical = cos_t * sin_x * sin_y * np.sin(2 * np.pi * (z_plane + 1 / 2) * beta)

    # for t in range(0, N):
    #     for i in range(0, M):
    #         for j in range(0, M):
    #             Ez_plane_analytical[t, i, j] = np.sin(2 * np.pi * i * beta) * np.sin(2 * np.pi * j * beta) * \
    #                                            np.sin(2 * np.pi * (z_plane + 1 / 2) * beta) * \
    #                                            np.cos(np.sqrt(12) * np.pi * c * t)
    # for i in range(0, M):
    #     for j in range(0, M):
    #         for k in range(0, M - 1):
    #             Ez[0, i, j, k] = np.sin(2 * np.pi * i * beta) * np.sin(2 * np.pi * j * beta) * \
    #                              np.sin(2 * np.pi * (k + 1 / 2) * beta)

    f = h5py.File('../Project/Ez_plane_a.h5', 'w')
    f.create_dataset('Ez_plane_a', data=Ez_plane_analytical)
    f.close()

    _, _, Ez, _, _, _ = yee_3d((M, M, M), N, initial_fields=(Ex, Ey, Ez, Bx, By, Bz))

    return Ex, Ey, Ez, Bx, By, Bz


if __name__ == '__main__':
    M = 50
    N = 200
    z_plane = M // 2

    # _, _, Ez, _, _, _ = initial_wave()
    # _, _, Ez, _, _, _ = initial_analytical(periods=1)
    _, _, Ez, _, _, _ = yee_3d((M, M, M), N, pulse=True)

    # Get and save value in plane
    Ez_plane = Ez[:, :, :, z_plane]

    f = h5py.File('../Project/Ez_plane.h5', 'w')
    f.create_dataset('Ez_plane', data=Ez_plane)
    f.close()

    # Plot solution at time t_plot
    # t_plot = 50

    # x = np.arange(M)
    # Y, X = np.meshgrid(x, x)
    #
    # plt.style.use('fivethirtyeight')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z[t_plot, :, :])
    #
    # plt.show()
