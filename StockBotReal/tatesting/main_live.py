#python 3.6.5
import pandas
import numpy as np
import scipy
import scipy.optimize

import sympy as sym
import matplotlib.pyplot as plt
import talib
from sklearn import preprocessing
x = pandas.DataFrame(np.linspace(0, 10, 11))
prices = pandas.read_csv("./datasets/IBM.csv")
closes = prices['Close']
closes = talib.SMA(closes)
print(prices)
print("############################")
print(len(closes))
def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def parabola3(x, a, b, c):
    return a*x**3 + b*x + c
params = [-0.1, 0.5, 10]
y3 = parabola3(x, *params)
print("y3")
#y3 = y3[0]
#x3 = 0
#list1 = [10, -21, 4, -45, 66, -93, 1]
x3 = [-2, -1, 0, 1, 2]
y3 = [-8, -1, 0, 1, 8]


def parabolavertex(a, b, c):
    vertex = [(-b / (2 * a)), (((4 * a * c) - (b * b)) / (4 * a))]
    print ("Vertex: (" , (-b / (2 * a)) , ", "
           ,(((4 * a * c) - (b * b)) / (4 * a)) , ")" )

    print ("Focus: (" , (-b / (2 * a)) , ", "
           , (((4 * a * c) - (b * b) + 1) / (4 * a)) , ")" )

    print ("Directrix: y="
           , (int)(c - ((b * b) + 1) * 4 * a ))

    return vertex


print(y3)
print("############################")
slices = closes[56:-1*(len(closes)-116)]
print(slices)

slices_x = slices.index.values.tolist()

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
#x3, y3 = [0, -5]

# Calculate the unknowns of the equation y=ax^2+bx+c
#a, b, c = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)


slices_y = slices.values.tolist()
for index, val in enumerate(slices_y):
    print(index)
    #slices_y[index] = round(val, 2)

#x_normalized = (slices_x - slices_x.mean())/slices_x.std()
print("############################")

print(slices_x)
print("############################")
print(slices_y)
print("############################")
x = [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
y = [125.48346000000001, 125.50963000000002, 125.53229666666668, 125.56696333333335, 125.60829666666667, 125.65163000000001, 125.68829666666667, 125.72879666666668, 125.76413000000001, 125.80113, 125.83546333333334, 125.87131666666667, 125.90409000000001, 125.93409, 125.96907333333334, 126.00278, 126.03695333333333, 126.06851333333333, 126.09696666666666, 126.12458999999998, 126.15266333333332, 126.17866333333333, 126.19617333333333, 126.21120666666667, 126.22320666666668, 126.23420666666668, 126.24520666666669, 126.25587333333335, 126.26613000000002, 126.26729666666668, 126.26763000000001, 126.26579666666667, 126.26165, 126.24931666666667, 126.23481666666667, 126.21581666666667, 126.19615, 126.17797333333334, 126.16314000000001, 126.14564000000001, 126.12664000000001, 126.10531666666667, 126.08554333333335, 126.06487666666668, 126.04204333333335, 126.01874333333335, 125.99140333333335, 125.96751000000002, 125.94275000000002, 125.91379333333336, 125.8875566666667, 125.86122333333336, 125.83871333333335, 125.81438000000001, 125.79138000000002, 125.77288, 125.75154666666666, 125.72904666666665, 125.71279, 125.69634333333333]

x = x[:-27]
y = y[:-27]
for index, val in enumerate(y):
    y[index] = round(val, 2)


#print(y)

fit_params, pcov = scipy.optimize.curve_fit(parabola, x,y)
print(fit_params)
vertex = parabolavertex(*fit_params)
plt.scatter(vertex[0], vertex[1])
x = np.array(x)
y_fit = parabola(np.array(x), *fit_params)
print(y_fit)

print(pcov)

plt.plot(x, y_fit)


plt.plot(x, y)
plt.show()
#print(x[2:-1 * (len(x) - 5 - 1)])
