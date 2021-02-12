import scipy.optimize
import sympy as sym
from sympy import *
import scipy.linalg as la
from scipy.signal import argrelextrema


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


from matplotlib import pyplot as plt
import numpy as np

params = [-0.1, 0.5, 10]
x = np.linspace(-10, 10, 31)
y = parabola(x, params[0], params[1], params[2])
print(y)
print(y.shape)
plt.plot(x, y, label='analytical')

plt.legend(loc='lower right')

r = np.random.RandomState(42)
y_with_errors = y + r.uniform(-1, 1, y.size)
plt.plot(x, y_with_errors, label='sample')

plt.legend(loc='lower right')

# plt.show()

fit_params, pcov = scipy.optimize.curve_fit(parabola, x, y_with_errors)
print("pcov")
print(pcov)
for param, fit_param in zip(params, fit_params):
    print(param, fit_param)
a, b = symbols('a, b')


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

    return A, B, C

    # Define your three known points
    x1, y1 = [2, 11]
    x2, y2 = [-4, 35]
    x3, y3 = [0, -5]

    # Calculate the unknowns of the equation y=ax^2+bx+c
    a, b, c = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)


def determinePosNeg(x, y):
    maxs = argrelextrema(np.array([x, y]), np.greater)
    print("Start determine parabolla")
    max = maxs[0]
    print(maxs)
    downval = 0
    upval = 0
    for val in x:

        print(val)
        if val <= max:
            print("less than max")
            downval = val if downval is not 0 else downval
        elif val >= max:
            print("val greater than max")
            upval = val if upval < val else upval
    if upval < max and downval < max:
        print("negative slope")
    elif upval > max and downval > max:
        print("positive slope")


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

    return A, B, C


# equations1 = Eq(6, a*(5+b)**2+10)
# equations2 = Eq(8, a*(8+b)**2+10)
y_fit = parabola(x, *fit_params)
# sol = solve([equations1, equations2], [a, b])
a, b, c = calc_parabola_vertex(x[1], y_fit[1], x[10], y_fit[10], x[20], y_fit[20])
print(f"y = {a}x^2+{b}x+{c}")
# print(sol)

plt.plot(x, y_fit, label='fit')
determinePosNeg(x, y_fit)
plt.plot()
plt.legend(loc='lower right')

plt.show()
