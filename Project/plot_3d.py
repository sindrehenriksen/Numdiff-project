import h5py
import numpy as np

from yee_method import yee_3d, initialize_fields_3d


def initial_wave():
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, N)
    l = np.arange(M // 2)
    wave = np.sin(l / l[-1] * np.pi)
    start_y = M // 4
    stop_y = M // 4 + M // 2
    Ez[0, M // 2, start_y:stop_y, M // 2] = wave
    By[0, M // 2, start_y:stop_y, M // 2] = wave

    _, _, Ez, _, _, _ = yee_3d(M, N, initial_fields=(Ex, Ey, Ez, Bx, By, Bz))

    return Ex, Ey, Ez, Bx, By, Bz


def initial_analytical_zero_boundary():
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, N)

    sin_x = np.sin(2 * np.pi * np.linspace(0, s, M)).reshape(M, 1, 1)
    sin_y = np.sin(2 * np.pi * np.linspace(0, s, M)).reshape(1, M, 1)
    # h = s / (M - 1)
    # sin_z = np.sin(2 * np.pi * np.linspace(h / 2, s - h / 2, M - 1)).reshape(1, 1, M - 1)
    Ez[0, :, :, :] = sin_x * sin_y #* sin_z

    cos_t = np.cos(8 * np.pi * c * np.linspace(0, t_end, N + 1)).reshape(N + 1, 1, 1)
    sin_x = sin_x.reshape(1, M, 1)
    sin_y = sin_y.reshape(1, 1, M)
    Ez_plane_analytical = cos_t * sin_x * sin_y #* np.sin(2 * np.pi * (z_plane + 1 / 2) * h)

    f = h5py.File('../Project/Ez_plane_a.h5', 'w')
    f.create_dataset('Ez_plane_a', data=Ez_plane_analytical)
    f.close()

    _, _, Ez, _, _, _ = yee_3d(M, N, initial_fields=(Ex, Ey, Ez, Bx, By, Bz))

    return Ex, Ey, Ez, Bx, By, Bz


def initial_analytical_periodic(*, type):
    x = np.linspace(0, s, M).reshape(M, 1, 1)
    y = np.linspace(0, s, M).reshape(1, M, 1)
    z = np.linspace(0, s, M - 1).reshape(1, 1, M - 1)

    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, N)

    if type == 'x':
        Ez[0, :, :, :] = np.sin(2 * np.pi * x)
        By_x = np.linspace(s / M / 2, s - s / M / 2, M - 1).reshape(M - 1, 1, 1)
        By[0, :, :, :] = np.sin(2 * np.pi * (By_x + c * t_end / (2 * M)))

    elif type == 'xy':
        Ez[0, :, :, :] = np.sin(2 * np.pi * (x + y))
        x_By = np.linspace(0, s, M - 1).reshape(M - 1, 1, 1)
        By[0, :, :, :] = -np.sin(2 * np.pi * ((x_By + y) - np.sqrt(2) * c * t_end /(2 * N)))

    elif type == 'xyz':
        Ez[0, :, :, :] = np.sin(2 * np.pi * (x + y + z))
        x_By = np.linspace(0, s, M - 1).reshape(M - 1, 1, 1)
        By[0, :, :, :] = np.sin(2 * np.pi * ((x_By + y + z) - np.sqrt(3) * c * t_end /(2 * N)))

    else:
        return ValueError

    t = np.linspace(0, t_end, N + 1).reshape(N + 1, 1, 1)
    x = x.reshape(1, M, 1)
    y = y.reshape(1, 1, M)

    if type == 'x':
        Ez_plane_analytical = np.sin(2 * np.pi * (x + c * t)) * np.ones(M).reshape(1, 1, M)

    elif type == 'xy':
        Ez_plane_analytical = np.sin(2 * np.pi * (x + y) - np.sqrt(2) * c * (2 * np.pi) * t)

    else:
        Ez_plane_analytical = np.sin(2 * np.pi * (x + y + z_plane) - np.sqrt(3) * c * (2 * np.pi) * t)


    f = h5py.File('../Project/Ez_plane_a.h5', 'w')
    f.create_dataset('Ez_plane_a', data=Ez_plane_analytical)
    f.close()

    _, _, Ez, _, _, _ = yee_3d(M, N, t_end=t_end, s=s, initial_fields=(Ex, Ey, Ez, Bx, By, Bz), boundary='periodic')

    return Ex, Ey, Ez, Bx, By, Bz


def initial_analytical_random(boundary):
    x = np.linspace(0, s, M).reshape(M, 1, 1)
    y = np.linspace(0, s, M).reshape(1, M, 1)
    z = np.linspace(0, s, M).reshape(1, 1, M)
    x_short = np.linspace(s / (M * 2), s - s / (M * 2), M - 1).reshape(M - 1, 1, 1)
    y_short = np.linspace(s / (M * 2), s - s / (M * 2), M - 1).reshape(1, M - 1, 1)
    z_short = np.linspace(s / (M * 2), s - s / (M * 2), M - 1).reshape(1, 1, M - 1)

    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, N)

    Ex[0, :, :, :] = np.sin(2 * np.pi * y) + z - x_short
    Ey[0, :, :, :] = np.cos(2 * np.pi * z) * z * 3 + x
    Ez[0, :, :, :] = np.sin(np.pi * x) + np.sin(7.5 * np.pi * x) / 7.4 - y * x + z_short

    _, _, Ez, _, _, _ = yee_3d(M, N, t_end=t_end, s=s, initial_fields=(Ex, Ey, Ez, Bx, By, Bz), boundary=boundary)

    return Ex, Ey, Ez, Bx, By, Bz


if __name__ == '__main__':
    M = 50
    N = 200
    z_plane = M // 2

    c = 299792458
    s = 1
    t_end = 1 / c

    # _, _, Ez, _, _, _ = initial_wave()
    _, _, Ez, _, _, _ = initial_analytical_zero_boundary()
    # _, _, Ez, _, _, _ = initial_analytical_periodic(type='x')
    # _, _, Ez, _, _, _ = initial_analytical_periodic(type='xy')
    # _, _, Ez, _, _, _ = initial_analytical_random(boundary='periodic')
    # _, _, Ez, _, _, _ = yee_3d(M, N, pulse=True, boundary='periodic')

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
