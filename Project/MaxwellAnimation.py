import numpy as np
from yee_method import yee_1d
import matplotlib.pyplot as plt
from matplotlib import animation
# import os
#
# os.environ['QT_API'] = 'pyqt'


class MaxwellAnimation:
    def __init__(self, M, N, t_end=None, x_end=None, initial_fields=None, boundary='pmc', pulse=False):
        '''
        :param M: Tuple of integers specifying amount of nodes in each spatial direction (simply an integer in 1D)
        :param initial_fields: Tuple of arrays containing field information necessary for running the Yee method.
        In 3 dimensions: (Ex, Ey, Ez, Bx, By, Bz), with specific shapes (asserted in Yee method function)
        '''

        self.M = M
        self.N = N
        self.t = 0
        self.t_end = t_end
        self.x_end = x_end
        self.boundary = boundary
        self.pulse = pulse

        if initial_fields is None:
            if isinstance(M, int):
                self.fields = (np.zeros((self.M)), np.zeros((self.M)))

                self.yee = yee_1d

            else:
                raise NotImplementedError

        else:
            self.fields = initial_fields

            if isinstance(M, int):
                assert self.fields[0].shape == self.fields[1].shape == (self.M,)
                self.yee = yee_1d

        self.animate()

    def animate(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(0, M), ylim=(-1, 1))
        line, = ax.plot([], [], lw=2)
        x = np.arange(self.M)
        t = np.arange(self.N + 1)

        def init():
            line.set_data(x, self.fields[0])
            return line,

        def animation_step(_):
            self.update_fields(1)
            y = self.fields[0]
            line.set_data(x, y)
            return line,

        step = max(self.N // 120, 1)
        anim = animation.FuncAnimation(fig, animation_step, frames=t[::step], init_func=init, interval=10, blit=True)

        plt.show()

    def update_fields(self, run_time_steps):
        self.fields = self.yee(self.M, self.N, t0=self.t, run_time_steps=run_time_steps, t_end=self.t_end,
                               x_end = self.x_end, initial_fields=self.fields, boundary=self.boundary, pulse=self.pulse)
        self.t += run_time_steps


if __name__ == '__main__':
    M = 100
    N = 100
    c = 299792458
    t_end = 1 / c
    x_end = 1

    mr_probz = MaxwellAnimation(M, M - 1, t_end=t_end, x_end=x_end, boundary='abc', pulse=True)

    # Ez = np.sin(2 * np.pi * np.linspace(0, x_end, M))
    # By = np.zeros(M)
    # By = -np.sin(2 * np.pi * np.linspace(0, x_end, M))
    # mr_probz = MaxwellAnimation(M, N, t_end=t_end, x_end=x_end, initial_fields=(Ez, By), boundary='periodic')

