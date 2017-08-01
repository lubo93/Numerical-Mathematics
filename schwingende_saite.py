import numpy as np
from scipy.linalg import lu, solve_triangular
import matplotlib.pyplot as plt

def power_method(A, n_rep = 2000):
    x = np.random.random(A.shape[0])
    x = x / np.linalg.norm(x)

    for k in range(n_rep):
        x = np.dot(A, x)
        x = x / np.linalg.norm(x)

    eigenvalue = np.dot(x.transpose(), np.dot(A, x))

    return eigenvalue

def inverse_power_method(A, n_rep = 100):
    P, L, U = lu(A)

    x = np.random.random(A.shape[0])
    x = x / np.linalg.norm(x)

    for k in range(n_rep):
        x = np.dot(P.transpose(), x)
        x = solve_triangular(L, x, lower=True)
        x = solve_triangular(U, x)
        x = x / np.linalg.norm(x)

    eigenvalue = np.dot(x.transpose(), np.dot(A, x))
    return eigenvalue
    
L = 2
N = 513
x = np.linspace(0, L, N+1)
h = x[1] - x[0]
A = np.diag(-2*np.ones(N-1)) + np.diag(np.ones(N-2), 1) + np.diag(np.ones(N-2), -1)
A /= h**2

print -1.0*inverse_power_method(A), -1.0*power_method(A)

ewh, evh = np.linalg.eigh(A)
ewh = -1.0*ewh
I = np.argsort(ewh)
ewh = ewh[I]
evh = evh[:,I]
print ewh[0], ewh[-1]

plt.figure()
for i in range(10):
    plt.plot(x[1:-1], evh[:,i], linewidth=1., label=r'$\psi_%d(x)$'%i)

plt.xlabel('$x$')
plt.ylabel('$\psi(x)$')
plt.legend(loc=4)
plt.tight_layout()
plt.show()