import numpy as np


def yee_1d(M, N, *, starting_time_step=0, run_time_steps=0, t_end=None, x_end=None, initial_fields=None, boundary='pmc',
           pulse=False):
    '''
    :param initial_fields: Tuple of arrays ((Ez, By)) with shapes (M,)
    '''

    if t_end is None and x_end is None:
        cp = 1
        t0 = starting_time_step

    elif t_end is not None and x_end is not None:
        c = 299792458
        h = x_end / (M - 1)
        k = t_end / N

        cp = c * k / h
        assert cp <= 1  # Courant number

        t0 = k * starting_time_step
    else:
        raise ValueError('Can\'t define only t_end or x_end.')

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
                Ez[M // 3] += np.exp(-((t + t0) - 30) ** 2 / 100)  # assuming k = 1
            By[:-1] = cp * (Ez[1:] - Ez[:-1]) + By[:-1]

    elif boundary == 'abc':  # Absorbing boundary condition
        if np.isclose(cp, 1):
            for t in range(1, run_time_steps + 1):
                Ez[0] = Ez[1]
                Ez[1:] = cp * (By[1:] - By[:-1]) + Ez[1:]

                if pulse:
                    Ez[M // 3] += np.exp(-((t + t0) - 30) ** 2 / 100)  # assuming k = 1

                By[-1] = By[-2]
                By[:-1] = cp * (Ez[1:] - Ez[:-1]) + By[:-1]
        else:
            raise NotImplementedError

    elif boundary == 'periodic':
        assert not pulse
        for t in range(1, run_time_steps + 1):
            Ez[1:] = cp * (By[1:] - By[:-1]) + Ez[1:]
            Ez[0] = Ez[-1]

            By[:-1] = cp * (Ez[1:] - Ez[:-1]) + By[:-1]
            By[-1] = By[-0]

    else:
        raise NotImplementedError

    return Ez, By


def yee_3d(M, N, *, t0=0, run_time_steps=0, t_end=None, s=None, initial_fields=None, boundary='pmc', pulse=False):
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
        assert cp <= 1 / np.sqrt(3)  # Courant number

    if initial_fields is None:
        Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, N)
    else:
        Ex, Ey, Ez, Bx, By, Bz = initial_fields
        assert Ex.shape == (N + 1, M - 1, M, M) and \
               Ey.shape == (N + 1, M, M - 1, M) and \
               Ez.shape == (N + 1, M, M, M - 1) and \
               Bx.shape == (N + 1, M, M - 1, M - 1) and \
               By.shape == (N + 1, M - 1, M, M - 1) and \
               Bz.shape == (N + 1, M - 1, M - 1, M)

    if run_time_steps == 0:
        run_time_steps = N

    if t0 != 0:
        index0 = (t0 * N) // t_end
    else:
        index0 = 0

    if boundary == 'pmc':  # Perfect magnetic (/electric) conductor
        for t in range(1, run_time_steps + 1):
            # print(t)
            update_E_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp, M)
            if pulse:
                Ez[index0 + t, 0, 1:-1, 1:-1] = np.exp(-((t + t0) - 30) ** 2 / 100)  # assuming k = 1
            update_B_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp, M)

    elif boundary == 'periodic':
        for t in range(1, run_time_steps + 1):
            # print(t)
            update_E_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp, M)
            if pulse:
                Ez[index0 + t, 0, :, :] = np.exp(-((t + t0) - 30) ** 2 / 100)  # assuming k = 1
                # Ez[t, -1, :, :] = 0
            update_B_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp, M)

    else:
        raise NotImplementedError

    return Ex, Ey, Ez, Bx, By, Bz


def initialize_fields_3d(M, N):
    Ex = np.zeros((N + 1, M - 1, M, M))
    Ey = np.zeros((N + 1, M, M - 1, M))
    Ez = np.zeros((N + 1, M, M, M - 1))
    Bx = np.zeros((N + 1, M, M - 1, M - 1))
    By = np.zeros((N + 1, M - 1, M, M - 1))
    Bz = np.zeros((N + 1, M - 1, M - 1, M))
    return Ex, Ey, Ez, Bx, By, Bz


def update_E_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp, M):
    Ex[t, :, 1:-1, 1:-1] = Ex[t - 1, :, 1:-1, 1:-1] + cp * \
                                                      ((Bz[t - 1, :, 1:, 1:-1] - Bz[t - 1, :, :-1, 1:-1]) -
                                                       (By[t - 1, :, 1:-1, 1:] - By[t - 1, :, 1:-1, :-1]))
    Ey[t, 1:-1, :, 1:-1] = Ey[t - 1, 1:-1, :, 1:-1] + cp * \
                                                      ((Bx[t - 1, 1:-1, :, 1:] - Bx[t - 1, 1:-1, :, :-1]) -
                                                       (Bz[t - 1, 1:, :, 1:-1] - Bz[t - 1, :-1, :, 1:-1]))
    Ez[t, 1:-1, 1:-1, :] = Ez[t - 1, 1:-1, 1:-1, :] + cp * \
                                                      ((By[t - 1, 1:, 1:-1, :] - By[t - 1, :-1, 1:-1, :]) -
                                                       (Bx[t - 1, 1:-1, 1:, :] - Bx[t - 1, 1:-1, :-1, :]))


def update_B_internal(Ex, Ey, Ez, Bx, By, Bz, t, cp, M):
    Bx[t, 1:-1, :, :] = Bx[t - 1, 1:-1, :, :] - cp * \
                                                ((Ez[t, 1:-1, 1:, :] - Ez[t, 1:-1, :-1, :]) -
                                                 (Ey[t, 1:-1, :, 1:] - Ey[t, 1:-1, :, :-1]))
    By[t, :, 1:-1, :] = By[t - 1, :, 1:-1, :] - cp * \
                                                ((Ex[t, :, 1:-1, 1:] - Ex[t, :, 1:-1, :-1]) -
                                                 (Ez[t, 1:, 1:-1, :] - Ez[t, :-1, 1:-1, :]))
    Bz[t, :, :, 1:-1] = Bz[t - 1, :, :, 1:-1] - cp * \
                                                ((Ey[t, 1:, :, 1:-1] - Ey[t, :-1, :, 1:-1]) -
                                                 (Ex[t, :, 1:, 1:-1] - Ex[t, :, :-1, 1:-1]))


def update_E_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp, M):
    Ex[t, :, :, :] = Ex[t - 1, :, :, ] + cp * \
                          ((np.concatenate((Bz[t - 1, :, :, :], Bz[t - 1, :, 0, :].reshape(M - 1, 1, M)), axis=1) -
                            np.concatenate((Bz[t - 1, :, -1, :].reshape(M - 1, 1, M), Bz[t - 1, :, :, :]), axis=1)) -
                           (np.concatenate((By[t - 1, :, :, :], By[t - 1, :, :, 0].reshape(M - 1, M, 1)), axis=2) -
                            np.concatenate((By[t - 1, :, :, -1].reshape(M - 1, M, 1), By[t - 1, :, :, :]), axis=2)))
    Ey[t, :, :, ] = Ey[t - 1, :, :, :] + cp * \
                          ((np.concatenate((Bx[t - 1, :, :, :], Bx[t - 1, :, :, 0].reshape(M, M - 1, 1)), axis=2) -
                            np.concatenate(((Bx[t - 1, :, :, -1]).reshape(M, M - 1, 1), Bx[t - 1, :, :, :]), axis=2)) -
                           (np.concatenate((Bz[t - 1, :, :, :], Bz[t - 1, 0, :, :].reshape(1, M - 1, M)), axis=0) -
                            np.concatenate((Bz[t - 1, -1, :, :].reshape(1, M - 1, M), Bz[t - 1, :, :, :]), axis=0)))
    Ez[t, :, :, :] = Ez[t - 1, :, :, :] + cp * \
                          ((np.concatenate((By[t - 1, :, :, :], By[t - 1, 0, :, :].reshape(1, M, M - 1)), axis=0) -
                            np.concatenate((By[t - 1, -1, :, :].reshape(1, M, M - 1), By[t - 1, :, :, :]), axis=0)) -
                           (np.concatenate((Bx[t - 1, :, :, :], Bx[t - 1, :, 0, :].reshape(M, 1, M - 1)), axis=1) -
                            np.concatenate((Bx[t - 1, :, -1, :].reshape(M, 1, M - 1), Bx[t - 1, :, :, :]), axis=1)))


def update_B_periodic(Ex, Ey, Ez, Bx, By, Bz, t, cp, M):
    Bx[t, :, :, :] = Bx[t - 1, :, :, :] - cp * \
                                          ((Ez[t, :, 1:, :] - Ez[t, :, :-1, :]) -
                                           (Ey[t, :, :, 1:] - Ey[t, :, :, :-1]))
    By[t, :, :, :] = By[t - 1, :, :, :] - cp * \
                                          ((Ex[t, :, :, 1:] - Ex[t, :, :, :-1]) -
                                           (Ez[t, 1:, :, :] - Ez[t, :-1, :, :]))
    Bz[t, :, :, :] = Bz[t - 1, :, :, :] - cp * \
                                          ((Ey[t, 1:, :, :] - Ey[t, :-1, :, :]) -
                                           (Ex[t, :, 1:, :] - Ex[t, :, :-1, :]))
