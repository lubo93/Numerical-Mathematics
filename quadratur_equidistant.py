from numpy import *
import matplotlib.pyplot as plt

f = lambda x: 1.0/sqrt(2*pi)*exp(-0.5*x**2) 

def mittelpunkt(f, a, b, N, h):
    res = 0
    for i in range(N):
        xi = a + i*h
        xip1 = a + (i+1)*h
        res += (xip1-xi)*f(0.5*(xi+xip1))
    return res

def trapez(f, a, b, N, h):
    res = 0
    for i in range(N):
        xi = a + i*h
        xip1 = a + (i+1)*h
        res += 0.5*(xip1-xi)*(f(xi)+f(xip1))
    return res

def simpson(f, a, b, N, h):
    res = 0
    for i in range(N):
        xi = a + i*h
        xip1 = a + (i+1)*h
        res += 1./6*(xip1-xi)*(f(xi)+4*f(0.5*(xi+xip1))+f(xip1))
    return res

a = -10
b = 10
N = 100
h = 1.0*(b-a)/N

int_err_mp = []
int_err_tra = []
int_err_simp = []
h_arr = []

for N in xrange(4,100):
    h = 1.0*(b-a)/N
    int_err_mp.append( abs(mittelpunkt(f, a, b, N, h)-1))
    int_err_tra.append( abs(trapez(f, a, b, N, h)-1))
    int_err_simp.append( abs(simpson(f, a, b, N, h)-1))
    h_arr.append(h)
    
plt.figure()
plt.title(r'Equidistant numerical integration', fontsize=26.)
plt.plot(h_arr, int_err_mp, label=r'Midpoint rule')
plt.plot(h_arr, int_err_tra, label=r'Trapezoidal rule')
plt.plot(h_arr, int_err_simp, label=r'Simpson rule')
plt.xlabel(r'Step size')
plt.ylabel(r'Error')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc=4)
plt.tight_layout()
plt.show()