from numpy import array, exp, linspace, asarray
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ode45 import ode45

l1 = 0.5
l2 = 0.1
PhiA0 = 1.0
PhiB0 = 0.5

g = lambda y: array([-l1*y[0], l1*y[0]-l2*y[1]])
Phi_A = lambda t: PhiA0*exp(-l1*t)
Phi_B = lambda t: (PhiB0-l1/(l2-l1)*PhiA0)*exp(-l2*t)+l1/(l2-l1)*PhiA0*exp(-l1*t)

def expleuler(g,t0,tend,y0,N):
    t, dt = linspace(t0,tend,N+1,retstep=True)
    y_ee = []
    y_ee.append(y0)
    for i in range(N):
        y_ee.append( y_ee[-1] + dt*g(y_ee[-1]) )        
    
    return t, asarray(y_ee)

def impleuler(g,t0,tend,y0,N):
    t, dt = linspace(t0,tend,N+1,retstep=True)
    y_ie = []
    y_ie.append(y0)
    for i in range(N):
        y_ie.append( fsolve(lambda y: y_ie[-1]+dt*g(y)-y, y_ie[-1]) )        
    
    return t, asarray(y_ie)

def mittelpkt(g,t0,tend,y0,N):
    t, dt = linspace(t0,tend,N+1,retstep=True)
    y_mp = []
    y_mp.append(y0)
    for i in range(N):
        y_mp.append( fsolve(lambda y: y_mp[-1]+dt*g(0.5*(y_mp[-1]+y))-y, y_mp[-1]) )        
    
    return t, asarray(y_mp)
    
N = 50
t0 = 0
tend = 15
y0 = array([PhiA0,PhiB0])

t_ode45, y_ode45 = ode45(lambda t,y: g(y), (t0,tend), y0)
t_ee, y_ee = expleuler(g,t0,tend,y0,N)
t_ie, y_ie = impleuler(g,t0,tend,y0,N)
t_mp, y_mp = mittelpkt(g,t0,tend,y0,N)

plt.figure()
plt.plot(t_ode45, Phi_A(t_ode45), color='k', linewidth=8., alpha=0.4, label=r'analytical')
plt.plot(t_ode45, Phi_B(t_ode45), color='k', linewidth=8., alpha=0.4)
plt.plot(t_ode45, y_ode45[:,0], color='b', marker='^', markersize=8., label=r'ode45')
plt.plot(t_ode45, y_ode45[:,1], color='b', marker='^', markersize=8.)
plt.plot(t_ee, y_ee[:,0], color='c', marker='o', markersize=8., label=r'expl. Euler')
plt.plot(t_ee, y_ee[:,1], color='c', marker='o', markersize=8.)
plt.plot(t_ie, y_ie[:,0], color='r', marker='s', markersize=8., label=r'impl. Euler')
plt.plot(t_ie, y_ie[:,1], color='r', marker='s', markersize=8.)
plt.plot(t_mp, y_mp[:,0], color='r', marker='v', markersize=8., label=r'midpoint')
plt.plot(t_mp, y_mp[:,1], color='r', marker='v', markersize=8.)
plt.xlabel('$t$')
plt.ylabel('$\Phi(t)$')
plt.xlim([0,15])
plt.legend(loc=1)
plt.tight_layout()
plt.show()