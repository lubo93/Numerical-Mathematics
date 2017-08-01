from numpy import *
from numpy.linalg import lstsq, norm
import matplotlib.pyplot as plt

def gauss_newton(x,F,J,tol):
    s = lstsq(J(x),F(x))[0]
    x = x-s
    while norm(s) > tol*norm(x):
        s = lstsq(J(x),F(x))[0]
        x = x-s
    return x
    

I1 = lambda t, x: x[0]+x[1]*exp(-x[2]*t)
I2 = lambda t, x: x[0]-x[1]*exp(-x[2]*t)

JI1 = lambda t, x: array([ones_like(t), 1.0*exp(-x[2]*t), -t*x[1]*exp(-x[2]*t)]).T
JI2 = lambda t, x: array([ones_like(t), -1.0*exp(-x[2]*t), t*x[1]*exp(-x[2]*t)]).T
t_arr = array([0, 5, 10, 15, 20, 25])
I1_arr = array([24.34, 18.93, 17.09, 16.27, 15.97, 15.91])
I2_arr = array([9.66, 18.8, 22.36, 24.07, 24.59, 24.91])

F1 = lambda x: I1(t_arr, x)-I1_arr
F2 = lambda x: I2(t_arr, x)-I2_arr

JF1 = lambda x: JI1(t_arr, x)
JF2 = lambda x: JI2(t_arr, x)

a10 = [10, 5, 0]
a20 = [30, 10, 0]

a1 = gauss_newton(a10, F1, JF1, 10**-3)
a2 = gauss_newton(a20, F2, JF2, 10**-3)

t = linspace(0,25,100)
plt.figure()
plt.plot(t, I1(t, a1), '-', linewidth=2.)
plt.plot(t, I2(t, a2), '-', linewidth=2.)
plt.plot(t_arr, I1_arr, 'o', label='$I_1$', markersize=10)
plt.plot(t_arr, I2_arr, 'o', label='$I_2$', markersize=10)
plt.xlabel('$t$')
plt.ylabel('$I(t)$')
plt.legend(loc=4)
plt.tight_layout()
plt.show()