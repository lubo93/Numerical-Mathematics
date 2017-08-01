from numpy import sin, cos, linspace, asarray, pi
import matplotlib.pyplot as plt

f = lambda phi: -9.81*sin(phi) 

def velocity_verlet(x0, v0, dt, N):
    x_vv = []
    v_vv = []
    x_vv.append(x0)
    v_vv.append(v0)
    for i in range(N):
        x_vv.append( x_vv[-1] + dt*v_vv[-1] + 0.5*dt**2 * f(x_vv[-1]) )
        v_vv.append( v_vv[-1] + 0.5*dt*( f(x_vv[-1]) + f(x_vv[-2]) ))
    
    return asarray(x_vv), asarray(v_vv)
    
N = 1000
T = 10
dt = 1.0*T/N

x0 = 0.5*pi
v0 = 0

t_arr = linspace(0,T,N+1)

# explicit euler + plot
x_vv, v_vv = velocity_verlet(x0, v0, dt, N)

plt.figure()
plt.title('Velocity Verlet', fontsize=26.)
plt.plot(t_arr, 0.5*v_vv**2+9.81*(1-cos(x_vv)), label='Etot')
plt.plot(t_arr, 0.5*v_vv**2, label='Ekin')
plt.plot(t_arr, 9.81*(1-cos(x_vv)), label='Epot')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc=3)
plt.tight_layout()
plt.show()