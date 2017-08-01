import numpy as np
import matplotlib.pyplot as plt

G = 1.0
M = 1.0

U = lambda r: -G*M / r

dUdr = lambda r: G*M / r**2

T = lambda p: 0.5*np.sum(p**2, axis=1)

V = lambda q: U(np.linalg.norm(q, axis=1))

H = lambda q, p: T(p) + V(q)

gradT = lambda p: p

gradV = lambda q: -1.0*dUdr(np.linalg.norm(q)) * q/np.linalg.norm(q)

def PhiT(y, dt):
    q, p = y[0,:], y[1,:]

    y1 = np.empty_like(y)
    y1[0,:] = q + dt*gradT(p)
    y1[1,:] = p

    return y1

def PhiV(y, dt):
    q, p = y[0,:], y[1,:]

    y1 = np.empty_like(y)
    y1[0,:] = q
    y1[1,:] = p + dt*gradV(q)

    return y1


def strang_splitting_step(Phi_a, Phi_b, y0, dt):
    y_a = Phi_a(y0, 0.5*dt)
    y_ba = Phi_b(y_a, dt)
    y_aba = Phi_a(y_ba, 0.5*dt)
    
    return y_aba
    
def integrate(Phi_a, Phi_b, y0, t_end, n_steps):

    t, dt = np.linspace(0, t_end, n_steps+1, retstep=True)
    y = []
    y.append(y0)
    for i in range(n_steps):
        y.append(strang_splitting_step(Phi_a, Phi_b, y[-1], dt))

    return t, np.asarray(y)


q = np.array([1.0, 0.0])
p = np.array([0.0, 1.2])

y = np.empty((2, 2))
y[0,:] = q
y[1,:] = p

t_end = 50.0
n_steps = 1000

t, y = integrate(PhiV, PhiT, y, t_end, n_steps)

plt.figure()
plt.plot(y[:,0,0], y[:,0,1])
plt.axis('equal')
plt.xlabel(r'$q_1$')
plt.ylabel(r'$q_2$')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, H(y[:,0,:], y[:,1,:]), label="H")
plt.plot(t, V(y[:,1,:]), label="V")
plt.plot(t, T(y[:,0,:]), label="T")
plt.xlabel(r'Time')
plt.ylabel(r'Energy')
plt.legend(loc="best")
plt.tight_layout()
plt.show()