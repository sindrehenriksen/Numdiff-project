from solutions import Solutions
from errors import Errors
import numpy as np
from yee_method import yee_1d
from yee_method import yee_3d
from sympy import *
from matplotlib import pyplot as plt

# An analytical solution for the one-dimensional case is obtained by initializing the class with
# desired initial conditions.

x, t, c = symbols("x t c")
speed_of_light = 299792458
f = sin(2 * pi * x)
g = - 2 * pi * speed_of_light * cos(2 * pi * x)
m = 200
n = 200
OneDimSolutions = Solutions(f, g, speed_of_light)
OneDimErrors = Errors(OneDimSolutions)
init_electric = OneDimSolutions.plot_analytical(0, 1, m, 1 / speed_of_light, get_field=True)
init_magnetic = OneDimSolutions.plot_analytical(0, 1, m, 1 / speed_of_light, get_field=True)
init_fields = init_electric, init_magnetic

# Obtaining convergence plot in 1D space using the classes. Initial conditions can changed as pleased

OneDimErrors.plot_error_space(4, 10, 1, 1 / (1 * speed_of_light))
OneDimErrors.plot_error_time(4, 10, 1, 1 / (1 * speed_of_light), ref_ana=False, expo=12, reference_order=2)

# Running other scripts to produce plots as in the report

import plot_convergence_1d
import plot_convergence_3d

# Plotting the numerical solution to a one-dimensional problem along with the analytical solution.

plt.plot(np.linspace(0, 1, m), yee_1d(m, n, t_end=1 / (1 * speed_of_light), x_end=1, initial_fields=init_fields)[0])
OneDimSolutions.plot_analytical(0, 1, m, 1/(1 * speed_of_light))
plt.show()

#The following script saves the data generated in Python in a format that can be read and plotted in Matlab.
#Running the script plot_surface.m in Matlab after running this script plots the data.

import plot_3d
