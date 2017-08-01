from numpy import zeros, cos, array, dot, asarray
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Magnus constants
c1 = 0.5*(1. - 0.5773502691896258)
c2 = 0.5*(1. + 0.5773502691896258)
a1 = 0.5*(0.5 - 0.5773502691896258)
a2 = 0.5*(0.5 + 0.5773502691896258)
# Parameters
tspan = [0,100]
eps = 0.25
omega = 1.
y0 = 1; y0p = 0
f = lambda t, omega, eps: omega**2 + eps*cos(t)

def A(t, omega, eps):
    res = zeros((2,2))
    res[0,1] = 1.
    res[1,0] = -f(t, omega, eps)
    return res

N = 10**4
u0 = array([y0,y0p])
t = [0.]
u = 1.*u0
uu = [1.*u]
h = (tspan[1]-tspan[0])/(1.0*N)
for k in range(N):
    t0 = k*h
    t1 = t0 + c1*h
    t2 = t0 + c2*h
    A1 = A(t1,omega,eps)
    A2 = A(t2,omega,eps)
    u = dot(expm(a1*h*A1+a2*h*A2),dot(expm(a2*h*A1+a1*h*A2),u))
    t += [t0+h]
    uu += [u]

plt.figure()
plt.plot(t, asarray(uu)[:,0], linewidth=2., label=r'Magnus Integration')
plt.xlim([0,100])
plt.ylim([-2,2])
plt.xlabel(r'$t$')
plt.ylabel(r'$f(t)$')
plt.legend(loc=3)
plt.tight_layout()
plt.show()