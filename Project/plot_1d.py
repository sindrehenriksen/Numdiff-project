import matplotlib.pyplot as plt
import numpy as np
from yee_method import yee_1d


def generate_time_domain_array_pulse(boundary):
    Ez = np.zeros((M, N + 1))
    Ez_t = np.zeros(M)
    By_t = np.zeros(M)

    for t in range(N):
        Ez_t, By_t = yee_1d(M, N, t0=t, run_time_steps=1, initial_fields=(Ez_t, By_t), boundary=boundary, pulse=True)
        Ez[:, t + 1] = Ez_t
    return Ez


def generate_time_domain_array_periodic(standing_wave=False):
    c = 299792458
    t_end = 1 / c
    x_end = 1

    Ez = np.zeros((M, N + 1))
    Ez_t = np.sin(2 * np.pi * np.linspace(0, x_end, M))
    if standing_wave:
        By_t = np.zeros(M)
    else:
        By_t = -np.sin(2 * np.pi * (np.linspace(0, x_end, M) + c * t_end / (2 * M)))

    Ez[:, 0] = Ez_t

    for t in range(N):
        Ez_t, By_t = yee_1d(M, N, t0=t, run_time_steps=1, initial_fields=(Ez_t, By_t), boundary='periodic')
        Ez[:, t + 1] = Ez_t
    return Ez


if __name__ == '__main__':
    M = 100
    N = 250

    type = 2

    if type == 1:
        Ez = generate_time_domain_array_pulse(boundary='pmc') * 2
        filename = 'figures/waterfall_pulse_pcm.pdf'
    elif type == 2:
        Ez = generate_time_domain_array_pulse(boundary='abc') * 2
        filename = 'figures/waterfall_pulse_abc.pdf'
    elif type == 3:
        Ez = generate_time_domain_array_periodic()
        filename = 'figures/waterfall_periodic_sinusoidal.pdf'
    elif type == 4:
        Ez = generate_time_domain_array_periodic(standing_wave=True)
        filename = 'figures/waterfall_standing_wave.pdf'
    else:
        raise ValueError

    waves = 30
    step = N // waves
    for i in range(0, N + 1, step):
        plt.plot(np.arange(M), i + Ez[:, i] * step, color='black', linewidth=0.5)

    plt.ylabel('t [temporal index]')
    plt.xlabel('x [spacial index]')

    plt.savefig(filename, bbox_inches='tight')
    plt.show()
