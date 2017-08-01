from numpy import *
from numpy.linalg import solve
import matplotlib.pyplot as plt

f = lambda y: 5*y*(1 - y)
fp = lambda y: 5 - 2*5*y
logistic = lambda x: 1./(1+9*exp(-5.*x))

# Rosenbrock-Wanner Ordnung 2
def ROW2_step(yi, h):
    
    yi = atleast_2d(yi)
    n = yi.shape[0]
    a = 1.0 / (2.0 + sqrt(2.0))
    I = identity(n)
    J = fp(yi)
    A = I - a*h*J
    # k1
    b1 = f(yi)
    k1 = solve(A, b1)
    # k2
    b2 = f(yi+0.5*h*k1) - a*h*dot(J,k1)
    k2 = solve(A, b2)
    # Advance
    yip1 = yi + h*k2
    return yip1

# Rosenbrock-Wanner Ordnung 3
def ROW3_step(yi, h):

    yi = atleast_2d(yi)
    n = yi.shape[0]
    a = 1.0 / (2.0 + sqrt(2.0))
    d31 = - (4.0 + sqrt(2.0)) / (2.0 + sqrt(2.0))
    d32 = (6.0 + sqrt(2.0)) / (2.0 + sqrt(2.0))
    I = identity(n)
    J = fp(yi)
    A = I - a*h*J
    # k1
    b1 = f(yi)
    k1 = solve(A, b1)
    # k2
    b2 = f(yi+0.5*h*k1) - a*h*dot(J,k1)
    k2 = solve(A, b2)
    # k3
    b3 = f(yi+h*k2) - d31*h*dot(J,k1) - d32*h*dot(J,k2)
    k3 = solve(A, b3)
    # Advance
    yip1 = yi + h/6.0*(k1 + 4*k2 + k3)
    return yip1


def solver(t0, y0, h, N, step):
    
    t_ROW = []
    y_ROW = []
    
    t_ROW.append(t0)
    y_ROW.append(y0)
    
    for i in range(1,N+1):
        t_ROW.append(t_ROW[-1]+h)
        y_ROW.append(step(y_ROW[-1], h))

    return asarray(t_ROW), asarray(y_ROW)

T = 3.0
N = 100
h = T/float(N)
t0 = 0
y0 = 0.1

t_ROW2, y_ROW2 = solver(t0, y0, h, N, ROW2_step)
t_ROW3, y_ROW3 = solver(t0, y0, h, N, ROW3_step)

plt.figure()
plt.title('Logistic Equation ROW', fontsize=26.)
plt.plot(t_ROW2, logistic(t_ROW2), label=r'exact')
plt.plot(t_ROW2, y_ROW2, 'o', label=r'ROW 2')
plt.plot(t_ROW3, y_ROW3, 'o', label=r'ROW 3')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.legend(loc=4)
plt.tight_layout()
plt.show()