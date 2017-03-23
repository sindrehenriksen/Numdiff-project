import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import matplotlib.animation as animation


class Solutions:
    x, t = symbols("x t")

    def __init__(self, init_displacement, init_velocity, constant):
        self.c = constant
        self.f = init_displacement
        #self.f = self.f.subs(c, self.c)
        self.g = init_velocity
        #self.g = self.g.subs(c, self.c)
        self.electric_field = self._analytical_electric()
        self.magnetic_field = self._analytical_magnetic()

    def _analytical_electric(self):
        x, t = symbols("x t")
        electric_field_part_1 = (self.f.subs(x, x - t * self.c) + self.f.subs(x, x + t * self.c)) / 2
        velocity_integral = integrate(self.g, (x, x - t * self.c, x + t * self.c))
        electric_field_part_2 = velocity_integral / (2 * self.c)
        return electric_field_part_1 + electric_field_part_2

    def _analytical_magnetic(self):
        x, t = symbols("x t")
        diff_electric = diff(self.electric_field, x)
        return integrate(diff_electric, t)

    def eval_analytical_electric(self, x_num, t_num):
        x, t = symbols("x t")
        return self.electric_field.subs([(x, x_num), (t, t_num)])

    def eval_analytical_magnetic(self, x_num, t_num):
        x, t = symbols("x t")
        return self.magnetic_field.subs([(x, x_num), (t, t_num)])

    def plot_analytical(self, start, stop, steps, time, electric=True, get_field=False):
        x_values = np.arange(start, stop, (stop - start) / steps)
        field = np.zeros(steps)
        for i in range(steps):
            if electric:
                field[i] = self.eval_analytical_electric(x_values[i], time)
            else:
                field[i] = self.eval_analytical_magnetic(x_values[i], time)
        if get_field:
            return field
        else:
            plt.plot(x_values, field)
            plt.show()
            return None

    def animation(self, x_start, x_stop, t_start, t_stop, steps, electric=True):
        t_values = np.arange(t_start, t_stop, (t_stop - t_start) / steps)
        x_values = np.arange(x_start, x_stop, (x_stop - x_start) / steps)
        fig, ax = plt.subplots()
        if electric:
            field = self.plot_analytical(x_start, x_stop, steps, t_values[0], electric=electric, get_field=True)
        elif electric == False:
            field = self.plot_analytical(x_start, x_stop, steps, t_values[steps // 2], electric=electric,
                                         get_field=True)
        line, = ax.plot(x_values, field)

        def animate(i):
            field = self.plot_analytical(x_start, x_stop, steps, t_values[i], electric=electric, get_field=True)
            line.set_ydata(field)
            return line,

        def init():
            line.set_ydata(np.ma.array(x_values, mask=True))
            return line,

        ani = animation.FuncAnimation(fig, animate, frames=steps, init_func=init, interval=5, blit=True)
        plt.show()
        #ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

if __name__ == "__main__":
    x, t, c = symbols("x t c")
    speed_of_light = 299792458
    f = exp(-(x - 30) ** 2 / 100)
    g = x*0
    problem = Solutions(f, g, speed_of_light)
    problem.plot_analytical(0, 80, 200, 10 / (1 * speed_of_light), electric=True)
#problem.animation(-70, 130, 0, 100 / speed_of_light, 50, electric=True)


# x, y, z, t, c = symbols("x y z t c")
# f = sin(pi*x) + cos(pi*y) + sin(pi*z)
# g = sin(pi*y)
# analytical_electric_3d(f, g)


