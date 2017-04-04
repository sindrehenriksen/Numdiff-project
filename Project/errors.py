import numpy as np
from yee_method import yee_1d
from sympy import *
from matplotlib import pyplot as plt
from solutions import Solutions


class Errors:
    x, t, c, = symbols("x t c")
    speed_of_light = 299792458

    def __init__(self, solutions):
        self.solutions = solutions

    def plot_error_space(self, start, end, x_end, t_end, reference_order=2, x_dir=True):
        points = end - start
        errors = np.zeros(points)
        step_vector = np.zeros(points)

        for i in range(points):
            print(i)
            if x_dir:
                m = 2 ** (start + i)
                n = 2 ** end
                h = x_end / m
            else:
                m = 2 ** start
                n = 2 ** (start + i)
                h = x_end / m
                k = t_end / n
            initial_field_electric = self.solutions.plot_analytical(0, x_end, m, 0, electric=True,
                                                                    get_field=True)
            initial_field_magnetic = self.solutions.plot_analytical(0, x_end, m, t_end / (2 * n), electric=False,
                                                                    get_field=True)
            fields_initial = initial_field_electric, initial_field_magnetic
            field_analytical = self.solutions.plot_analytical(0, x_end, m, t_end, electric=True, get_field=True)
            field_numerical = yee_1d(m, n, t_end=t_end, x_end=x_end, initial_fields=fields_initial,
                                     boundary="periodic")[0]
            if x_dir:
                step_vector[i] = h
            else:
                step_vector[i] = k
            errors[i] = np.sqrt(h) * np.linalg.norm(field_numerical - field_analytical)
            # plt.plot(np.linspace(0, x_end, m), field_numerical)
            # plt.plot(np.linspace(0, x_end, m), field_analytical)
            # plt.show()

        plt.loglog(step_vector, errors, "-o")
        if reference_order is not None:
            plt.loglog(step_vector, 2 * step_vector ** reference_order, linestyle=":")
            # plt.loglog(step_vector, 0.1 * step_vector ** 1, linestyle=":")
            plt.legend(('Error', 'Order 2'), loc="best")
        print(np.log(errors[points - 1] / errors[0]) / np.log(step_vector[points - 1] / step_vector[0]))
        plt.ylabel('Error, 2 - norm')
        if x_dir:
            plt.xlabel('Stepsize [m]')
        else:
            plt.xlabel('Temporal step size [m]')
        # plt.title(function)
        plt.show()

    def plot_error_time(self, start, end, x_end, t_end, t_dir=True, reference_order=1, ref_ana=True, expo=10):
        points = end - start
        errors = np.zeros(points)
        step_vector = np.zeros(points)
        reference_initial_electric = self.solutions.plot_analytical(0, x_end, 2 ** start, 0, electric=True,
                                                                    get_field=True)
        reference_initial_magnetic = self.solutions.plot_analytical(0, x_end, 2 ** start, 0, electric=False,
                                                                    get_field=True)
        reference_initial = reference_initial_electric, reference_initial_magnetic
        reference_field_full = np.zeros(2 ** expo)
        for p in range(2 ** expo):
            dummy = yee_1d(2 ** start, 2 ** expo, t0=p * t_end / 2 ** expo, run_time_steps=1, t_end=t_end,
                           x_end=x_end, initial_fields=reference_initial, boundary="periodic")
            reference_field_full[p] = dummy[0][2 ** start - 1]
            reference_initial = dummy
        for i in range(points):
            n = 2 ** (start + i)
            m = 2 ** start
            k = t_end / n
            end_field = np.zeros(n)
            initial_field_electric = np.copy(self.solutions.plot_analytical(0, x_end, m, 0, electric=True,
                                                                            get_field=True))
            initial_field_magnetic = np.copy(self.solutions.plot_analytical(0, x_end, m, 0, electric=False,
                                                                            get_field=True))
            fields_initial = initial_field_electric, initial_field_magnetic
            if ref_ana:
                reference_field = self.solutions.electric_along_time(0, t_end, n, x_end)
            else:
                reference_field = reference_field_full[::2 ** expo / n]
            for j in range(n):
                field_numerical = yee_1d(m, n, t0=j * t_end / n, run_time_steps=1, t_end=t_end,
                                         x_end=x_end, initial_fields=fields_initial, boundary="periodic")
                end_field[j] = field_numerical[0][m - 1]
                fields_initial = field_numerical
            step_vector[i] = k
            errors[i] = np.sqrt(k) * np.linalg.norm(end_field - reference_field)
            #plt.plot(np.linspace(0, t_end, n), end_field)
            #plt.plot(np.linspace(0, t_end, n), reference_field)
            #plt.show()

        plt.loglog(step_vector, errors, "-o")
        if reference_order is not None:
            plt.loglog(step_vector, 10e10 * step_vector ** reference_order, linestyle=":")
            plt.loglog(step_vector, 1 * step_vector ** 1, linestyle=":")
            plt.legend(('Error', 'Order 2', 'Order 1'), loc="best")
        print(np.log(errors[points - 1] / errors[0]) / np.log(step_vector[points - 1] / step_vector[0]))
        plt.ylabel('Error, 2 - norm')
        plt.xlabel('Temporal step size')
        plt.show()


if __name__ == "__main__":
    speed_of_light = 299792458
    x, t, c, = symbols("x t c")
    f = exp(-(x-30)**2/100)
    g = 0
    #g = 2 * pi * speed_of_light * sin(2 * pi * x)
    analytical = Solutions(f, g, speed_of_light)
    # analytical.animation(0, 100, 20 / speed_of_light, 80 / speed_of_light, 100, electric=True)
    # print(analytical.electric_field)
    print(analytical.magnetic_field)
    error = Errors(analytical)
    # error.plot_error_space(4, 12, 1, 1 / (1 * speed_of_light), x_dir=False)
    error.plot_error_time(4, 12, 1, 1 / (1 * speed_of_light), ref_ana=False, expo=14, reference_order=2)
