import matplotlib.pyplot as plt
import numpy as np

from yee_method import yee_1d

c = 299792458
t_end = 1 / c
x_end = 1


def analytical_1d(x, t):
    return np.sin(2 * np.pi * (x - c * t))

start = 3
stop = 14
N_const = 2 ** stop
M_const = 2 ** start
h_M_const = 1 / (M_const - 1)

Ez0_M_const = np.sin(2 * np.pi * np.linspace(0, x_end, M_const))
By0_M_const = -np.sin(2 * np.pi * (np.linspace(0, x_end, M_const) + c * t_end / (2 * M_const)))
ref_num_M_const = yee_1d(M_const, 2 ** (stop + 1), t_end=t_end, x_end=x_end,
                         initial_fields=(np.copy(Ez0_M_const), np.copy(By0_M_const)), boundary='periodic')[0]

error_N_const = np.zeros(stop - start + 1)
error_M_const = np.zeros(stop - start + 1)
h = 1 / (np.logspace(start, stop, stop - start + 1, base=2) - 1)  # unit x_end
k = 1 / np.logspace(start, stop, stop - start + 1, base=2)  # unit t_end
x_M_const = np.linspace(0, x_end, M_const)

for i in range(start, stop + 1):
    # Constant N
    M = 2 ** i
    Ez_N_const = np.sin(2 * np.pi * np.linspace(0, x_end, M))
    By_N_const = -np.sin(2 * np.pi * (np.linspace(0, x_end, M) + c * t_end / (2 * M)))
    numerical_1d_N_const = yee_1d(M, N_const, t_end=t_end, x_end=x_end,
                                  initial_fields=(Ez_N_const, By_N_const), boundary='periodic')[0]

    xi = np.linspace(0, x_end, 2 ** i)
    error_vec_N_const = numerical_1d_N_const - analytical_1d(xi, t_end)
    error_N_const[i - start] = np.sqrt(h[i - start]) * np.linalg.norm(error_vec_N_const)

    # Constant M
    N = 2 ** i
    numerical_1d_M_const = yee_1d(M_const, N, t_end=t_end, x_end=x_end,
                             initial_fields=(np.copy(Ez0_M_const), np.copy(By0_M_const)), boundary='periodic')[0]

    error_vec_M_const = numerical_1d_M_const - ref_num_M_const
    error_M_const[i - start] = np.sqrt(h_M_const) * np.linalg.norm(error_vec_M_const)

plt.loglog(h[:-1], error_N_const[:-1], '-o', label='Space')
plt.loglog(k, error_M_const, '-o', label='Time')
plt.loglog(h, h, '--', label='Order 1')
plt.loglog(h, h ** 2, '--', label='Order 2')

plt.legend(loc="best")
plt.ylabel('Error (2-norm)')
plt.xlabel('Stepsize (space: [m], time: [1/c])')

plt.savefig('figures/convergence_1d.pdf', bbox_inches='tight')
plt.show()
