from numpy import sqrt, array, arange, asarray
import matplotlib.pyplot as plt

f = lambda x: 24*(2./sqrt(x[0]**2+x[1]**2)**14-1./sqrt(x[0]**2+x[1]**2)**8)*x

def stoermer_verlet(x0, v0, dt, T):
    x_sv = []
    x_sv.append(x0)
    x1 = x0 + dt*v0 + 0.5*dt**2*f(x0)
    x_sv.append(x1)
    for i in range(int(T/dt)):
        x_sv.append( -x_sv[-2]+2*x_sv[-1]+dt**2*f(x_sv[-1]) )        
    
    return asarray(x_sv)
    
T = 15
dt = 0.02
v0 = array([1,0])

plt.figure()
for b in arange(0.15,3.15,0.15):
    x0 = array([-10,b])

    # stoermer-verlet + plot
    x_sv = stoermer_verlet(x0, v0, dt, T)
    plt.plot(x_sv[:,0], x_sv[:,1])

plt.title(r'Stoermer-Verlet', fontsize=26.)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=2)
plt.tight_layout()
plt.show()