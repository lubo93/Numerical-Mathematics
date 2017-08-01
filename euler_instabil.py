from numpy import linspace, asarray
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

f = lambda y, lamb: lamb*y
g_impl = lambda y, y0, lamb, dt: y0+dt*lamb*y-y

def explicit_euler(y0, lamb, dt, N):
    y_ee = []
    y_ee.append(y0)
    for i in range(N):
        y_ee.append( y_ee[-1] + dt*f(y_ee[-1],lamb) )        
    
    return asarray(y_ee)

def implicit_euler(y0, lamb, dt, N):
    y_ie = []
    y_ie.append(y0)
    for i in range(N):
        y_ie.append( fsolve(g_impl, y_ie[-1],args=(y_ie[-1],lamb,dt)) )        
    
    return asarray(y_ie)
       
N = 1000
T = 10
lamb = -10
dt = 1.0*T/N
y0 = 1

t_arr = linspace(0,T,N+1)

# explicit euler + plot
y_ee = explicit_euler(y0, lamb, dt, N)
print y_ee[0:10]

plt.figure()
plt.title(r'Stability Explicit Euler', fontsize=26.)
plt.plot(t_arr, y_ee, label=r'$\lambda=%d$, $h=%.2f$'%(lamb,dt))
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.legend(loc=2)
plt.tight_layout()
plt.show()