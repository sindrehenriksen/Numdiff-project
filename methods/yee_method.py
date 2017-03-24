import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def yee_1d(M, N, *, t0=0, run_time_steps=0, t_end=None, x_end=None, initial_fields=None, boundary='pmc', pulse=False):
    '''
    :param initial_fields: Tuple of arrays ((Ez, By)) with shapes (M,)
    '''

    if t_end is None or x_end is None:
        cp = 1
    else:
        c = 299792458
        h = x_end / (M - 1)
        k = t_end / N

        cp = c * k / h
        assert cp <= 1  # Courant number

    if initial_fields is None:
        Ez = np.zeros(M)
        By = np.zeros(M)
    else:
        Ez, By = initial_fields
        assert Ez.shape == By.shape == (M,)

    if run_time_steps == 0:
        run_time_steps = N

    if boundary == 'pmc':  # Perfect magnetic (/electric) conductor
        for t in range(1, run_time_steps + 1):
            Ez[1:] = cp * (By[1:] - By[:-1]) + Ez[1:]
            if pulse:
                Ez[M // 3] += np.exp(-((t + t0) - 30) ** 2 / 100)
            By[:-1] = cp * (Ez[1:] - Ez[:-1]) + By[:-1]

    elif boundary == 'abc':
        if np.isclose(cp, 1):
            for t in range(1, run_time_steps + 1):
                Ez[0] = Ez[1]
                Ez[1:] = cp * (By[1:] - By[:-1]) + Ez[1:]

                if pulse:
                    Ez[M // 3] += np.exp(-((t + t0) - 30) ** 2 / 100)

                By[-1] = By[-2]
                By[:-1] = cp * (Ez[1:] - Ez[:-1]) + By[:-1]
        else:
            raise NotImplementedError

    elif boundary == 'periodic':
        for t in range(1, run_time_steps + 1):
            Ez[1:] = cp * (By[1:] - By[:-1]) + Ez[1:]
            Ez[0] = Ez[-1]

            if pulse:
                Ez[M // 3] += np.exp(-((t + t0) - 30) ** 2 / 100)

            By[:-1] = cp * (Ez[1:] - Ez[:-1]) + By[:-1]
            By[-1] = By[-0]

    else:
        raise NotImplementedError

    return Ez, By


def yee_3d(M, N, t0=0, *, t_end=None, s=None, initial_fields=None, boundary='pmc', pulse=False):
    '''
    :param initial_fields: Tuple of arrays ((Ex, Ey, Ez, Bx, By, Bz)) with shapes as defined in assertion below
    '''

    if t_end is None or s is None:
        cp = 1 / np.sqrt(3)
    else:
        c = 299792458
        h = s / (M - 1)
        k = t_end / N

        cp = c * k / h
        assert cp <= 1  # Courant number

    if initial_fields is None:
        Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M)
    else:
        Ex, Ey, Ez, Bx, By, Bz = initial_fields
        assert Ex.shape == (N + 1, M - 1, M, M) and \
               Ey.shape == (N + 1, M, M - 1, M) and \
               Ez.shape == (N + 1, M, M, M - 1) and \
               Bx.shape == (N + 1, M, M - 1, M - 1) and \
               By.shape == (N + 1, M - 1, M, M - 1) and \
               Bz.shape == (N + 1, M - 1, M - 1, M)

    if boundary == 'pmc':
        for t in range(1, N + 1):
            print(t)
            update_E_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp)
            if pulse:
                Ez[t, 0, 1:-1, M // 2] = np.exp(-((t + t0) - 30) ** 2 / 100)
            update_B_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp)

    elif boundary == 'periodic':
        assert not pulse
        for t in range(1, N + 1):
            print(t)
            update_E_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp)
            update_B_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp)

    else:
        raise NotImplementedError

    return Ex, Ey, Ez, Bx, By, Bz


def initialize_fields_3d(M):
    Ex = np.zeros((N + 1, M - 1, M, M))
    Ey = np.zeros((N + 1, M, M - 1, M))
    Ez = np.zeros((N + 1, M, M, M - 1))
    Bx = np.zeros((N + 1, M, M - 1, M - 1))
    By = np.zeros((N + 1, M - 1, M, M - 1))
    Bz = np.zeros((N + 1, M - 1, M - 1, M))
    return Ex, Ey, Ez, Bx, By, Bz


def update_E_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp):
    Ex[t, :, 1:-1, 1:-1] = Ex[t - 1, :, 1:-1, 1:-1] + cp * \
                                                      ((Bz[t - 1, :, 1:, 1:-1] - Bz[t - 1, :, :-1, 1:-1]) -
                                                       (By[t - 1, :, 1:-1, 1:] - By[t - 1, :, 1:-1, :-1]))
    Ey[t, 1:-1, :, 1:-1] = Ey[t - 1, 1:-1, :, 1:-1] + cp * \
                                                      ((Bx[t - 1, 1:-1, :, 1:] - Bx[t - 1, 1:-1, :, :-1]) -
                                                       (Bz[t - 1, 1:, :, 1:-1] - Bz[t - 1, :-1, :, 1:-1]))
    Ez[t, 1:-1, 1:-1, :] = Ez[t - 1, 1:-1, 1:-1, :] + cp * \
                                                      ((By[t - 1, 1:, 1:-1, :] - By[t - 1, :-1, 1:-1, :]) -
                                                       (Bx[t - 1, 1:-1, 1:, :] - Bx[t - 1, 1:-1, :-1, :]))


def update_B_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp):
    Bx[t, 1:-1, :, :] = Bx[t - 1, 1:-1, :, :] - cp * \
                                                ((Ez[t, 1:-1, 1:, :] - Ez[t, 1:-1, :-1, :]) -
                                                 (Ey[t, 1:-1, :, 1:] - Ey[t, 1:-1, :, :-1]))
    By[t, :, 1:-1, :] = By[t - 1, :, 1:-1, :] - cp * \
                                                ((Ex[t, :, 1:-1, 1:] - Ex[t, :, 1:-1, :-1]) -
                                                 (Ez[t, 1:, 1:-1, :] - Ez[t, :-1, 1:-1, :]))
    Bz[t, :, :, 1:-1] = Bz[t - 1, :, :, 1:-1] - cp * \
                                                ((Ey[t, 1:, :, 1:-1] - Ey[t, :-1, :, 1:-1]) -
                                                 (Ex[t, :, 1:, 1:-1] - Ex[t, :, :-1, 1:-1]))


def update_E_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp):
    # x-axis
    Ey[t, 0, :, 1:-1] = Ey[t - 1, 0, :, 1:-1] + cp * \
                                                ((Bx[t - 1, 0, :, 1:] - Bx[t - 1, 0, :, :-1]) -
                                                 (Bz[t - 1, 0, :, 1:-1] - Bz[t - 1, -1, :, 1:-1]))
    Ey[t, -1, :, 1:-1] = Ey[t - 1, -1, :, 1:-1] + cp * \
                                                  ((Bx[t - 1, -1, :, 1:] - Bx[t - 1, -1, :, :-1]) -
                                                   (Bz[t - 1, 0, :, 1:-1] - Bz[t - 1, -1, :, 1:-1]))
    Ez[t, 0, 1:-1, :] = Ez[t - 1, 0, 1:-1, :] + cp * \
                                                ((By[t - 1, 0, 1:-1, :] - By[t - 1, -1, 1:-1, :]) -
                                                 (Bx[t - 1, 0, 1:, :] - Bx[t - 1, 0, :-1, :]))
    Ez[t, -1, 1:-1, :] = Ez[t - 1, -1, 1:-1, :] + cp * \
                                                  ((By[t - 1, 0, 1:-1, :] - By[t - 1, -1, 1:-1, :]) -
                                                   (Bx[t - 1, -1, 1:, :] - Bx[t - 1, -1, :-1, :]))

    # y-axis
    Ex[t, :, 0, 1:-1] = Ex[t - 1, :, 0, 1:-1] + cp * \
                                                ((Bz[t - 1, :, 0, 1:-1] - Bz[t - 1, :, -1, 1:-1]) -
                                                 (By[t - 1, :, 0, 1:] - By[t - 1, :, 0, :-1]))
    Ex[t, :, -1, 1:-1] = Ex[t - 1, :, -1, 1:-1] + cp * \
                                                  ((Bz[t - 1, :, 0, 1:-1] - Bz[t - 1, :, -1, 1:-1]) -
                                                   (By[t - 1, :, -1, 1:] - By[t - 1, :, -1, :-1]))
    Ez[t, 1:-1, 0, :] = Ez[t - 1, 1:-1, 0, :] + cp * \
                                                ((By[t - 1, 1:, 0, :] - By[t - 1, :-1, 0, :]) -
                                                 (Bx[t - 1, 1:-1, 0, :] - Bx[t - 1, 1:-1, -1, :]))
    Ez[t, 1:-1, -1, :] = Ez[t - 1, 1:-1, -1, :] + cp * \
                                                  ((By[t - 1, 1:, -1, :] - By[t - 1, :-1, -1, :]) -
                                                   (Bx[t - 1, 1:-1, 0, :] - Bx[t - 1, 1:-1, -1, :]))

    # z-axis
    Ex[t, :, 1:-1, 0] = Ex[t - 1, :, 1:-1, 0] + cp * \
                                                ((Bz[t - 1, :, 1:, 0] - Bz[t - 1, :, :-1, 0]) -
                                                 (By[t - 1, :, 1:-1, 0] - By[t - 1, :, 1:-1, -1]))
    Ex[t, :, 1:-1, -1] = Ex[t - 1, :, 1:-1, -1] + cp * \
                                                  ((Bz[t - 1, :, 1:, -1] - Bz[t - 1, :, :-1, -1]) -
                                                   (By[t - 1, :, 1:-1, 0] - By[t - 1, :, 1:-1, -1]))
    Ey[t, 1:-1, :, 0] = Ey[t - 1, 1:-1, :, 0] + cp * \
                                                ((Bx[t - 1, 1:-1, :, 0] - Bx[t - 1, 1:-1, :, 0]) -
                                                 (Bz[t - 1, 1:, :, 0] - Bz[t - 1, :-1, :, -1]))
    Ey[t, 1:-1, :, -1] = Ey[t - 1, 1:-1, :, -1] + cp * \
                                                  ((Bx[t - 1, 1:-1, :, -1] - Bx[t - 1, 1:-1, :, -1]) -
                                                   (Bz[t - 1, 1:, :, 0] - Bz[t - 1, :-1, :, -1]))

    update_E_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp)

    # Periodic along x-axis, special case where E(a, b, c) should equal E(a, b', c')
    # Ex[t, :, 0, :] = Ex[t, :, 1, :]
    # Ex[t, :, -1, :] = Ex[t, :, -2, :]
    # Ez[t, :, 0, :] = Ez[t, :, 1, :]
    # Ez[t, :, -1, :] = Ez[t, :, -2, :]
    #
    # Ex[t, :, :, 0] = Ex[t, :, :, 1]
    # Ex[t, :, :, -1] = Ex[t, :, :, -2]
    # Ey[t, :, :, 0] = Ey[t, :, :, 1]
    # Ey[t, :, :, -1] = Ey[t, :, :, -2]


def update_B_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp):
    Bx[t, 0, :, :] = Bx[t - 1, 0, :, :] - cp * \
                                                ((Ez[t, 0, 1:, :] - Ez[t, 0, :-1, :]) -
                                                 (Ey[t, 0, :, 1:] - Ey[t, 0, :, :-1]))
    Bx[t, -1, :, :] = Bx[t - 1, -1, :, :] - cp * \
                                                ((Ez[t, -1, 1:, :] - Ez[t, -1, :-1, :]) -
                                                 (Ey[t, -1, :, 1:] - Ey[t, -1, :, :-1]))

    By[t, :, 0, :] = By[t - 1, :, 0, :] - cp * \
                                                ((Ex[t, :, 0, 1:] - Ex[t, :, 0, :-1]) -
                                                 (Ez[t, 1:, 0, :] - Ez[t, :-1, 0, :]))
    By[t, :, -1, :] = By[t - 1, :, -1, :] - cp * \
                                                ((Ex[t, :, -1, 1:] - Ex[t, :, -1, :-1]) -
                                                 (Ez[t, 1:, -1, :] - Ez[t, :-1, -1, :]))

    Bz[t, :, :, 0] = Bz[t - 1, :, :, 0] - cp * \
                                                ((Ey[t, 1:, :, 0] - Ey[t, :-1, :, 0]) -
                                                 (Ex[t, :, 1:, 0] - Ex[t, :, :-1, 0]))
    Bz[t, :, :, -1] = Bz[t - 1, :, :, -1] - cp * \
                                                ((Ey[t, 1:, :, -1] - Ey[t, :-1, :, -1]) -
                                                 (Ex[t, :, 1:, -1] - Ex[t, :, :-1, -1]))

    update_B_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp)


def initial_wave():
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M)
    l = np.arange(M // 2)
    wave = np.sin(l / l[-1] * np.pi)
    start_y = M // 4
    stop_y = M // 4 + M // 2
    Ez[0, M // 2, start_y:stop_y, M // 2] = wave
    Bx[0, M // 2, start_y:stop_y, M // 2] = wave

    _, _, Ez, _, _, _ = yee_3d(M, N, initial_fields=(Ex, Ey, Ez, Bx, By, Bz))

    return Ex, Ey, Ez, Bx, By, Bz


def initial_analytical(periods=1):
    c = 299792458
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M)
    x_end = 1

    sin_x = np.sin(2 * np.pi * np.linspace(0, x_end, M)).reshape(M, 1, 1)
    sin_y = np.sin(2 * np.pi * np.linspace(0, x_end, M)).reshape(1, M, 1)
    h = x_end / (M - 1)
    sin_z = np.sin(2 * np.pi * np.linspace(h / 2, x_end - h / 2, M - 1)).reshape(1, 1, M - 1)
    Ez[0, :, :, :] = sin_x * sin_y * sin_z

    t_end = 1 / c
    cos_t = np.cos(np.sqrt(12) * np.pi * c * np.linspace(0, t_end, N + 1)).reshape(N + 1, 1, 1)
    sin_x = sin_x.reshape(1, M, 1)
    sin_y = sin_y.reshape(1, 1, M)
    Ez_plane_analytical = cos_t * sin_x * sin_y * np.sin(2 * np.pi * (z_plane + 1 / 2) * h)

    f = h5py.File('../Project/Ez_plane_a.h5', 'w')
    f.create_dataset('Ez_plane_a', data=Ez_plane_analytical)
    f.close()

    _, _, Ez, _, _, _ = yee_3d(M, N, initial_fields=(Ex, Ey, Ez, Bx, By, Bz))

    return Ex, Ey, Ez, Bx, By, Bz


def initial_analytical_sigrid():
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M)
    s = 1
    x = np.linspace(0, s, M).reshape(M, 1, 1)
    y = np.linspace(0, s, M).reshape(1, M, 1)
    z = np.linspace(0, s, M - 1).reshape(1, 1, M - 1)

    Ez[0, :, :, :] = np.sin(2 * np.pi * (x + y + z))

    # Ez[0, :, :, :] = np.sin(2 * np.pi * x)
    # By_x = np.linspace(s / M / 2, s - s / M / 2, M - 1).reshape(M - 1, 1, 1)
    # By[0, :, :, :] = np.sin(2 * np.pi * By_x)

    c = 299792458
    t_end = 1 / c
    t = np.linspace(0, t_end, N + 1).reshape(N + 1, 1, 1)
    x = x.reshape(1, M, 1)
    y = y.reshape(1, 1, M)

    Ez_plane_analytical = np.sin(2 * np.pi * (x + y + z_plane) - np.sqrt(3) * c / (2 * np.pi) * t)

    # Ez_plane_analytical = np.sin(2 * np.pi * x + 2 * np.pi * c * t) * np.ones(M).reshape(1, 1, M)

    f = h5py.File('../Project/Ez_plane_a.h5', 'w')
    f.create_dataset('Ez_plane_a', data=Ez_plane_analytical)
    f.close()

    _, _, Ez, _, _, _ = yee_3d(M, N, t_end=t_end, s=s, initial_fields=(Ex, Ey, Ez, Bx, By, Bz), boundary='periodic')

    return Ex, Ey, Ez, Bx, By, Bz


if __name__ == '__main__':
    M = 50
    N = 200
    z_plane = M // 2

    # _, _, Ez, _, _, _ = initial_wave()
    # _, _, Ez, _, _, _ = initial_analytical(periods=1)
    _, _, Ez, _, _, _ = initial_analytical_sigrid()
    # _, _, Ez, _, _, _ = yee_3d(M, N, pulse=True)

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
