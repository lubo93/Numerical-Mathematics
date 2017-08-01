import numpy as np

f = lambda x: 1.0/(1+5*x**2)

# Knoten und Gewichte
def golub_welsch(n):
    
    i = np.arange(n-1)
    b = (i+1.0) / np.sqrt(4.0*(i+1)**2 - 1.0)
    J = np.diag(b, -1) + np.diag(b, 1)
    x, ev = np.linalg.eigh(J)
    w = 2 * ev[0,:]**2

    return x, w

# Nichtzusammengesetzte Gauss-Legendre Quadratur
def gauss_legendre(f, a, b, n):

    x_ref, w_ref = golub_welsch(n)

    x = a + (x_ref + 1.0)*(b - a)*0.5
    w = 0.5*(b - a)*w_ref

    return np.sum(w*f(x))

# Zusammengesetzte Gauss-Legendre Quadratur
def composite_legendre(f, a, b, N, n):

    I = 0.0

    dx = (b - a)/N
    for i in range(N):
        I += gauss_legendre(f, a + i*dx, a + (i+1)*dx, n)

    return I
    
print(composite_legendre(f, 0., 1., 10, 10))