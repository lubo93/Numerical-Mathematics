from numpy import array, asarray, linspace
from scipy.special import airy
import matplotlib.pyplot as plt

f = lambda t,y: array([y[1], t*y[0]])
    
def RK4(t0, y0, N, T):
    y_RK4 = []
    y_RK4.append(y0)
    t_RK4, h = linspace(t0, T, N+1, retstep=True)

    for i in range(N):
        k1 = f(t_RK4[i], y_RK4[-1])
        k2 = f(t_RK4[i]+0.5*h, y_RK4[-1]+0.5*h*k1)
        k3 = f(t_RK4[i]+0.5*h, y_RK4[-1]+0.5*h*k2)
        k4 = f(t_RK4[i]+h, y_RK4[-1]+h*k3)
        
        y_RK4.append(y_RK4[-1]+h*(1./6*k1+1./3*k2+1./3*k3+1./6*k4))
    
    return t_RK4, asarray(y_RK4)
    

# RK4 + plot

N = 500
T = -40
y0 = array([ 0.35502805388781723926,
            -0.25881940379280679841])
t0 = 0           
t_RK4, y_RK4 = RK4(t0, y0, N, T)
exact = asarray(airy(t_RK4))[0,:]



plt.figure()
plt.title(r'Airy function RK4', fontsize=26.)
plt.plot(t_RK4, y_RK4[:,0], 'o',label=r'RK4')
plt.plot(t_RK4, exact, label=r'exact')
plt.xlabel(r't')
plt.ylabel(r'Ai(t)')
plt.legend(loc=3)
plt.tight_layout()
plt.show()