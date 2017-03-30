import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from yee_method import yee_1d


class MaxwellAnimation:
    def __init__(self, M, N, t_end=None, x_end=None, initial_fields=None, boundary='pmc', pulse=False, analytical=None,
                 save=False):
        '''
        :param M: Tuple of integers specifying amount of nodes in each spatial direction (simply an integer in 1D)
        :param initial_fields: Tuple of arrays containing field information necessary for running the Yee method.
        In 3 dimensions: (Ex, Ey, Ez, Bx, By, Bz), with specific shapes (asserted in Yee method function)
        '''

        self.M = M
        self.N = N
        self.time_step = 0
        self.t_end = t_end
        self.x_end = x_end
        self.boundary = boundary
        self.pulse = pulse
        self.analytical = analytical
        self.yee = yee_1d
        self.save = save

        if initial_fields is None:
                self.fields = (np.zeros((self.M)), np.zeros((self.M)))
        else:
            self.fields = initial_fields
            assert self.fields[0].shape == self.fields[1].shape == (self.M,)

        self.animate()

    def animate(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(0, M), ylim=(-1, 1))
        line_num, = ax.plot([], [], lw=3)

        if self.analytical is not None:
            line_ana, = ax.plot([], [], lw=3, ls='--')

        x_indices = np.arange(self.M)
        t_indices = np.arange(self.N + 1)

        def init():
            line_num.set_data(x_indices, self.fields[0])

            if self.analytical is not None:
                return line_num, line_num

            return line_num,

        def animation_step(i):
            # current_fields = np.copy(self.fields)  # For reference solution
            self.update_fields(step)
            y = self.fields[0]
            line_num.set_data(x_indices, y)

            if self.analytical is not None:
                line_ana.set_data(x_indices, self.analytical(x_end * x_indices / (M - 1), t_end * t_indices[i] / N))

                # For reference solution
                # line_ana.set_data(x_indices,
                #    self.yee(self.M, self.N*1000, starting_time_step=self.time_step, run_time_steps=1000, t_end=self.t_end,
                #    x_end=self.x_end, initial_fields=current_fields, boundary=self.boundary, pulse=self.pulse)[0])

                return line_num, line_ana

            return line_num,

        step = max(self.N // 120, 1)
        anim = animation.FuncAnimation(fig, animation_step, frames=t_indices[::step], init_func=init, interval=10, blit=True)

        if self.save:
            anim.save('movies/wave.mp4', fps=24)

        plt.show()

    def update_fields(self, run_time_steps):
        self.fields = self.yee(self.M, self.N, starting_time_step=self.time_step, run_time_steps=run_time_steps,
                               t_end=self.t_end, x_end = self.x_end, initial_fields=self.fields, boundary=self.boundary,
                               pulse=self.pulse)
        self.time_step += run_time_steps


def numerical_source_pulse():
    MaxwellAnimation(M, N, boundary='pmc', pulse=True)


def analytial_sinusoidal(standing=False):
    Ez0 = np.sin(2 * np.pi * np.linspace(0, x_end, M))
    By0 = -np.sin(2 * np.pi * (np.linspace(0, x_end, M) + c * t_end / (2 * M)))

    if standing:
        By0 = np.zeros(M)
        analytical = None
    else:
        def analytical(x, t):
            return np.sin(2 * np.pi * (x - c * t))

    MaxwellAnimation(M, N, t_end=t_end, x_end=x_end, initial_fields=(Ez0, By0), boundary='periodic',
                                analytical=analytical)


def analytical_pulse(M, pulses=1):
    M = M + 1
    x = np.arange(M)
    middle_index = M / 2 - 2
    Ez0 = np.exp(-(x - middle_index) ** 2 / M)

    middle = x_end / 2
    if pulses == 1:
        Bz0 = -np.exp(-(x - middle_index) ** 2 / M)

        def analytical(x, t):
            return np.exp(-(-c * t + x - middle) ** 2 * (M - 1))
    elif pulses == 2:
        Bz0 = np.zeros(M)

        def analytical(x, t):
            return np.exp(-(c * t + x - middle) ** 2 * (M - 1)) / 2 + np.exp(-(-c * t + x - middle) ** 2 * (M - 1)) / 2

    MaxwellAnimation(M, N, t_end=t_end, x_end=1, initial_fields=(Ez0, Bz0), analytical=analytical)


if __name__ == '__main__':
    M = 100
    N = 100
    c = 299792458
    t_end = 1 / c
    x_end = 1

    # numerical_source_pulse()
    # analytial_sinusoidal()
    # analytial_sinusoidal(standing=True)
    analytical_pulse(M, pulses=1)
    # analytical_pulse(M, pulses=2)
