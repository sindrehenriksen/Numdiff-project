import matplotlib.pyplot as plt
import numpy as np

from yee_method import yee_3d, initialize_fields_3d


def analytical_3d(x, t, M):
    return np.sin(2 * np.pi * (x + c * t)) * np.ones(M).reshape(1, M, 1) * np.ones(M - 1).reshape(1, 1, M - 1)

c = 299792458
t_end = 1 / c
s = 1

# Constant N
start_M = 2
stop_M = 7
N_const = 2 ** (stop_M + 1)

error_N_const = np.zeros(stop_M - start_M + 1)
h = 1 / (np.logspace(start_M, stop_M, stop_M - start_M + 1, base=2) - 1)  # unit s

for i in range(start_M, stop_M + 1):
    M = 2 ** i

    Ex, Ey, Ez, Bx, By, Bz = initialize_fields_3d(M, N_const)
    xi = np.linspace(0, s, M).reshape(M, 1, 1)
    Ez[0, :, :, :] = np.sin(2 * np.pi * xi)
    x_By = np.linspace(s / (M * 2), s - s / (M * 2), M - 1).reshape(M - 1, 1, 1)
    By[0, :, :, :] = np.sin(2 * np.pi * (x_By + c * t_end / (2 * M)))

    _, _, Ez, _, _, _ = yee_3d(M, N_const, t_end=t_end, s=s,
                                    initial_fields=(Ex, Ey, Ez, Bx, By, Bz), boundary='periodic')

    error_mat_N_const = Ez[-1, :, :, :] - analytical_3d(xi, t_end, M)
    error_N_const[i - start_M] = (h[i - start_M]) ** (3 / 2) * np.linalg.norm(error_mat_N_const)

# Constant M
start_N = 3
stop_N = 9
M_const = 2 ** (start_N - 1)
h_M_const = 1 / (M_const - 1)

Ex0, Ey0, Ez0, Bx0, By0, Bz0 = initialize_fields_3d(M_const, 2 ** (stop_N + 1))
x_M_const = np.linspace(0, s, M_const).reshape(M_const, 1, 1)
x_M_const_short = np.linspace(s / (M_const * 2), s - s / (M_const * 2), M_const - 1).reshape(M_const - 1, 1, 1)
Ez0[0, :, :, :] = np.sin(2 * np.pi * x_M_const)
By0[0, :, :, :] = np.sin(2 * np.pi * (x_M_const_short + c * t_end / (2 * M_const)))

_, _, Ez_ref, _, _, _ = yee_3d(M_const, 2 ** (stop_N + 1), t_end=t_end, s=s,
                               initial_fields=(Ex0, Ey0, Ez0, Bx0, By0, Bz0), boundary='periodic')
Ez_ref_t_end = Ez_ref[-1, :, :, :]

error_M_const = np.zeros(stop_N - start_N + 1)
k = 1 / (np.logspace(start_N, stop_N, stop_N - start_N + 1, base=2) - 1)  # unit t_end

for i in range(start_N, stop_N + 1):
    N = 2 ** i
    Ex0, Ey0, Ez0, Bx0, By0, Bz0 = initialize_fields_3d(M_const, N)
    Ez0[0, :, :, :] = np.sin(2 * np.pi * x_M_const)
    By0[0, :, :, :] = np.sin(2 * np.pi * (x_M_const_short + c * t_end / (2 * M_const)))

    _, _, Ez, _, _, _ = yee_3d(M_const, N, t_end=t_end, s=s,
                               initial_fields=(Ex0, Ey0, Ez0, Bx0, By0, Bz0), boundary='periodic')

    error_mat_N_const = Ez[-1, :, :, :] - Ez_ref_t_end
    error_M_const[i - start_N] = (h_M_const) ** (3 / 2) * np.linalg.norm(error_mat_N_const)

start_stepsize = min(start_M, start_N)
stop_stepsize = max(stop_M, stop_N)
stepsize_range = 1 / (np.logspace(start_stepsize, stop_stepsize, stop_stepsize - start_stepsize + 1, base=2) - 1)

plt.loglog(h, error_N_const, '-o', label='Space')
plt.loglog(k, error_M_const, '-o', label='Time')
plt.loglog(stepsize_range, stepsize_range, '--', label='Order 1')
plt.loglog(stepsize_range, 2 * stepsize_range ** 2, '--', label='Order 2')

plt.legend(loc="best")
plt.ylabel('Error (Frobenius norm)')
plt.xlabel('Stepsize (space: [m], time: [1/c])')

plt.savefig('figures/convergence_3d.pdf', bbox_inches='tight')
plt.show()
