from numpy import *
import matplotlib.pyplot as plt

f = lambda x, y: x**2*y**2

def trapez(f, a, b, N):
    x, h = linspace(a, b, N+1, retstep=True)

    I = h*(0.5*f(x[0]) + sum(f(xx) for xx in x[1:-1]) + 0.5*f(x[-1]))

    return I
    
def trapez2d(f, a, b, Nx, c, d, Ny):
    F = lambda y: trapez(lambda x: f(x, y), a, b, Nx)

    return trapez(F, c, d, Ny)

h_arr = []
int_err_tra = []

for N in range(4,128):
    trapez_approx = trapez2d(f, 0, 1.0, N, 0, 1.0, N)
    h_arr.append(1./N)
    int_err_tra.append(abs(trapez_approx-1./9))

plt.figure()
plt.title(r'Trapezoidal integration in 2D', fontsize=26.)
plt.plot(h_arr, int_err_tra)
plt.plot(h_arr, array(h_arr)**2)mon
plt.xlabel(r'Step size')
plt.ylabel(r'Error')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc=4)
plt.tight_layout()
plt.show()