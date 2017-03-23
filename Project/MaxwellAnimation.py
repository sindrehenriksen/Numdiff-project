import numpy as np
from yee_method import yee_1d
import matplotlib.pyplot as plt
from matplotlib import animation
import os

os.environ['QT_API'] = 'pyqt'


class MaxwellAnimation:
    def __init__(self, M, N, initial_fields=None, boundary='pmc'):
        '''
        :param M: Tuple of integers specifying amount of nodes in each spatial direction (simply an integer in 1D)
        :param initial_fields: Tuple of arrays containing field information necessary for running the Yee method.
        In 3 dimensions: (Ex, Ey, Ez, Bx, By, Bz), with specific shapes (asserted in Yee method function)
        '''

        self.M = M
        self.N = N
        self.t = 0

        if initial_fields is None:
            if isinstance(M, int):
                self.fields = (np.zeros((self.M + 1)), np.zeros((self.M + 1)))

                self.yee = yee_1d

            else:
                raise NotImplementedError

        else:
            self.fields = initial_fields

            if isinstance(M, int):
                assert self.fields.shape
                self.yee = yee_1d

        self.boundary = boundary

        self.animate()

    def animate(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(0, M), ylim=(-1, 1))
        line, = ax.plot([], [], lw=2)
        x = np.arange(self.M + 1)
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

    def update_fields(self, time_steps):
        self.fields = self.yee(self.M, time_steps, self.t, self.fields, self.boundary, pulse=True)
        self.t += time_steps


if __name__ == '__main__':
    M = 200
    N = 10
    mr_probz = MaxwellAnimation(M, N)
