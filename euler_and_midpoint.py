from numpy import sin, cos, array, linspace, asarray
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ode45 import ode45

f = lambda alpha, p: array([p, -9.81*sin(alpha)])
f_ode45 = lambda t,y: array([y[1], -9.81*sin(y[0])])
g_impl = lambda x, y0, dt: y0+dt*f(x[0],x[1])-x
g_mid = lambda x, y0, dt: y0+dt*f(0.5*(x[0]+y0[0]),0.5*(x[1]+y0[1]))-x

def explicit_euler(y0, dt, N):
    y_ee = []
    y_ee.append(y0)
    for i in range(N):
        y_ee.append( y_ee[-1] + dt*f(y_ee[-1][0],y_ee[-1][1]) )        
    
    return asarray(y_ee)

def implicit_euler(y0, dt, N):
    y_ie = []
    y_ie.append(y0)
    for i in range(N):
        y_ie.append( fsolve(g_impl, y_ie[-1],args=(y_ie[-1],dt)) )        
    
    return asarray(y_ie)

def midpoint(y0, dt, N):
    y_mp = []
    y_mp.append(y0)
    for i in range(N):
        y_mp.append( fsolve(g_mid, y_mp[-1],args=(y_mp[-1],dt)) )        
    
    return asarray(y_mp)
    
N = 1000
T = 10
dt = 1.0*T/N

y0 = array([0.75, 0])

t_arr = linspace(0,T,N+1)

# explicit euler + plot
y_ee = explicit_euler(y0, dt, N)

plt.figure()
plt.title('Explicit Euler', fontsize=26.)
plt.plot(t_arr, 0.5*y_ee[:,1]**2+9.81*(1-cos(y_ee[:,0])), label='Etot')
plt.plot(t_arr, 0.5*y_ee[:,1]**2, label='Ekin')
plt.plot(t_arr, 9.81*(1-cos(y_ee[:,0])), label='Epot')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc=2)
plt.tight_layout()
plt.show()

# implicit euler + plot
y_ie = implicit_euler(y0, dt, N)

plt.figure()
plt.title('Implicit Euler', fontsize=26.)
plt.plot(t_arr, 0.5*y_ie[:,1]**2+9.81*(1-cos(y_ie[:,0])), label='Etot')
plt.plot(t_arr, 0.5*y_ie[:,1]**2, label='Ekin')
plt.plot(t_arr, 9.81*(1-cos(y_ie[:,0])), label='Epot')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc=1)
plt.tight_layout()
plt.show()

# midpoint + plot
y_mp = midpoint(y0, dt, N)

plt.figure()
plt.title('Midpoint', fontsize=26.)
plt.plot(t_arr, 0.5*y_mp[:,1]**2+9.81*(1-cos(y_mp[:,0])), label='Etot')
plt.plot(t_arr, 0.5*y_mp[:,1]**2, label='Ekin')
plt.plot(t_arr, 9.81*(1-cos(y_mp[:,0])), label='Epot')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc=3)
plt.tight_layout()
plt.show()

# ode45 + plot
t_ode45, y_ode45 = ode45(f_ode45, (0.0, T), y0)

plt.figure()
plt.title('ODE 45', fontsize=26.)
plt.plot(t_ode45, 0.5*y_ode45[:,1]**2+9.81*(1-cos(y_ode45[:,0])), label='Etot')
plt.plot(t_ode45, 0.5*y_ode45[:,1]**2, label='Ekin')
plt.plot(t_ode45, 9.81*(1-cos(y_ode45[:,0])), label='Epot')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc=3)
plt.tight_layout()
plt.show()