from numpy import tile, reshape, exp, pi, real, hstack, arange, cos, zeros, polyfit, log10, linspace, r_
from scipy import ifft
import matplotlib.pyplot as plt

# Runge-Funktion Definition
f = lambda x: 1.0/(1 + x**2)

# Clenshaw Algorithmus
def clenshaw(a,x):
    # Polynomgrad
    n = a.shape[0] - 1
    d = tile( reshape(a,n+1,1), (x.shape[0], 1) )
    d = d.T

    for j in range(n, 1, -1):
        d[j-1,:] = d[j-1,:] + 2.0*x*d[j,:]
        d[j-2,:] = d[j-2,:] - d[j,:]

    y = d[0,:] + x*d[1,:]

    return y

# Chebyshev Interpolation
def chebexp(y):
    # Polynomgrad
    n = y.shape[0] - 1
    
    # RHS Vektor
    t = arange(0, 2*n+2)
    z = exp(-pi*1.0j*n/(n+1.0)*t) * hstack([y,y[::-1]])

    # Loese lineares System
    c = ifft(z)

    # Berechne b_j
    t = arange(-n, n+2)
    b = real(exp(0.5j*pi/(n+1.0)*t) * c)

    # Berechne a_j
    a = hstack([ b[n], 2*b[n+1:2*n+1] ])

    return a

# Anzahl stuetzstellen
n = 30
# Berechne Chebychev Stuetzstellen
x_cheby = cos((2.*r_[0:n+1]+1.)/(n+1)*pi/2.)
# f an Chebychev Stuetzstellen
fx_cheby = f(x_cheby)
# berechne Chebychev Koeffizienten
a = chebexp(fx_cheby)

# Stuetzstellen und Funktionswerte zur Fehleranalyse
N  = 1000
xx  = linspace(-1,1,N)
fxx = f(xx)

# Funktionswerte entsprechend Clenshaw-Alogrithmus
y = clenshaw(a,x_cheby)

plt.figure()
plt.plot(xx,fxx, label=r'Runge function')
plt.plot(x_cheby, y, '-o', label=r'Chebychev interpolation')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.legend(loc=4)
plt.tight_layout()
plt.show()

# Fehler fuer verschiedene n
n_max = 200
err = zeros(n_max-1)
for n in range(2,n_max+1):
    x_cheby = cos((2.*r_[0:n+1]+1.)/(n+1)*pi/2.)
    fx_cheby = f(x_cheby)
    a = chebexp(fx_cheby)
    p = clenshaw(a,xx)

    # berechne Fehler
    err[n-2] = max(abs(fxx - p))

# exponentielle Konvergenz?
plt.semilogy(range(2,n_max+1),err,'b-')
plt.xlabel(r'Anzahl Knoten $n$')
plt.ylabel(r'Fehler')
plt.tight_layout()
plt.show()

# Fit exponentielle Konvergenz
pfit = polyfit(range(2,51),log10(err[2:51]),1)
print 'Exponetielle Konvergenz: %d**%f' % (10,pfit[0])

